"""Reconnect backoff, and what a dropped socket does to a call already sent."""

import asyncio

import pytest

from mcp_blender import blender_client, wire
from mcp_blender import server as server_module
from mcp_blender.blender_client import (
    RECONNECT_BASE_DELAY,
    RECONNECT_MAX_DELAY,
    BlenderClient,
    BlenderConnectionError,
    reconnect_delay,
)
from tests.test_client import MockBlenderServer


@pytest.fixture(autouse=True)
def token(monkeypatch, tmp_path):
    """Each test gets its own config dir with a token in it, and no ambient override."""
    monkeypatch.delenv("MESHWIRE_TOKEN", raising=False)
    monkeypatch.setenv("MESHWIRE_CONFIG_DIR", str(tmp_path / "meshwire"))
    return wire.ensure_token()


@pytest.fixture
async def mock_server():
    async with MockBlenderServer() as server:
        yield server


def test_reconnect_delay_is_jittered_and_capped():
    """The wait grows, two draws at one step differ, and nothing exceeds the cap."""
    low = reconnect_delay(1, 0.0)
    high = reconnect_delay(1, 1.0)
    assert low == pytest.approx(RECONNECT_BASE_DELAY * 0.5)
    assert high == pytest.approx(RECONNECT_BASE_DELAY)
    assert low != high
    assert reconnect_delay(2, 1.0) == pytest.approx(
        min(RECONNECT_MAX_DELAY, RECONNECT_BASE_DELAY * 2)
    )
    assert reconnect_delay(3, 1.0) > reconnect_delay(2, 1.0)
    assert reconnect_delay(30, 1.0) == pytest.approx(RECONNECT_MAX_DELAY)
    assert reconnect_delay(30, 0.0) == pytest.approx(RECONNECT_MAX_DELAY * 0.5)
    assert reconnect_delay(30, 0.3) <= RECONNECT_MAX_DELAY


@pytest.mark.asyncio
async def test_stopped_blender_is_not_contacted_on_every_call(monkeypatch, mock_server):
    """After a refusal, another call within the wait does not open a socket.

    Once that wait has passed the next call may dial again, and a success clears
    the wait so a later failure starts from the short delay.
    """
    clock = {"now": 1_000.0}
    monkeypatch.setattr(blender_client.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(blender_client.random, "random", lambda: 1.0)

    allow = {"on": False}
    opens = []
    real_open = asyncio.open_connection

    async def gated(host, port, *args, **kwargs):
        opens.append(port)
        if not allow["on"]:
            raise ConnectionRefusedError
        return await real_open(host, port, *args, **kwargs)

    monkeypatch.setattr(blender_client.asyncio, "open_connection", gated)
    client = BlenderClient(mock_server.host, mock_server.port)

    with pytest.raises(BlenderConnectionError):
        await client.connect(timeout=1.0)
    assert opens == [mock_server.port]

    with pytest.raises(BlenderConnectionError, match="jitter") as info:
        await client.connect(timeout=1.0)
    assert "capped at" in str(info.value)
    assert opens == [mock_server.port]

    clock["now"] += reconnect_delay(1, 1.0) / 2
    with pytest.raises(BlenderConnectionError, match="capped at"):
        await client.connect(timeout=1.0)
    assert opens == [mock_server.port]

    clock["now"] += reconnect_delay(1, 1.0)
    allow["on"] = True
    await client.connect(timeout=1.0)
    assert client.connected
    assert opens == [mock_server.port, mock_server.port]

    await client.disconnect()
    allow["on"] = False
    with pytest.raises(BlenderConnectionError):
        await client.connect(timeout=1.0)
    assert len(opens) == 3
    # The success cleared the streak, so this wait is the first-failure delay.
    clock["now"] += reconnect_delay(1, 1.0)
    allow["on"] = True
    await client.connect(timeout=1.0)
    assert len(opens) == 4
    await client.disconnect()


@pytest.mark.asyncio
async def test_repeated_calls_reuse_the_client_so_backoff_holds(monkeypatch, mock_server):
    """The server keeps one client. A new one per call would dial a stopped Blender every time."""
    previous = server_module._client
    server_module._client = None
    clock = {"now": 1_000.0}
    monkeypatch.setattr(blender_client.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(blender_client.random, "random", lambda: 1.0)
    opens = []

    async def refuse(host, port, *args, **kwargs):
        opens.append(port)
        raise ConnectionRefusedError

    monkeypatch.setattr(blender_client.asyncio, "open_connection", refuse)
    try:
        with pytest.raises(BlenderConnectionError):
            await server_module.get_client(mock_server.host, mock_server.port)
        with pytest.raises(BlenderConnectionError, match="capped at"):
            await server_module.get_client(mock_server.host, mock_server.port)
        assert opens == [mock_server.port]
    finally:
        client = server_module._client
        server_module._client = previous
        if client is not None:
            await client.disconnect()


@pytest.mark.asyncio
async def test_blender_restart_mid_session_reconnects(mock_server):
    """Blender quits and comes back on the same port; the next call reconnects."""
    mock_server.set_response("scene_info", {"name": "Scene"})
    client = BlenderClient(mock_server.host, mock_server.port)
    await client.connect()
    try:
        first = await client.send_command("scene_info")
        assert first["name"] == "Scene"
        hellos = len(mock_server.hellos)
        await mock_server.drop_clients()
        for _ in range(100):
            if not client.connected:
                break
            await asyncio.sleep(0.01)
        assert not client.connected

        second = await client.send_command("scene_info")
        assert second["name"] == "Scene"
        assert len(mock_server.hellos) == hellos + 1
        assert [command["method"] for command in mock_server.commands] == [
            "scene_info",
            "scene_info",
        ]
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_call_in_flight_when_the_socket_drops_is_not_retried(mock_server):
    """A scene-changing call already sent is not sent again when the socket drops.

    The error says the call may already have run in Blender. A later, different
    call may connect again; the lost one stays lost.
    """
    client = BlenderClient(mock_server.host, mock_server.port)
    await client.connect()
    try:
        mock_server.drop_command = True
        with pytest.raises(BlenderConnectionError) as info:
            await client.send_command("object_create", {"type": "cube"}, timeout=5.0)
        message = str(info.value)
        assert "object_create" in message
        assert "may already have run in Blender and was not retried" in message
        assert [command["method"] for command in mock_server.commands] == ["object_create"]
        assert len(mock_server.hellos) == 1
        assert not client.connected

        mock_server.drop_command = False
        result = await client.send_command("ping", timeout=5.0)
        assert result["pong"] is True
        assert [command["method"] for command in mock_server.commands] == [
            "object_create",
            "ping",
        ]
        assert len(mock_server.hellos) == 2
    finally:
        await client.disconnect()


# The add-on's idle timeout, as the client sees it.


@pytest.mark.parametrize(
    "value", [None, True, "300", -1, 0, float("inf"), float("nan"), [300], {"s": 300}]
)
def test_only_a_positive_finite_idle_timeout_counts(value):
    assert BlenderClient._advertised_idle(value) == 0.0


def test_a_positive_idle_timeout_is_kept():
    assert BlenderClient._advertised_idle(300) == 300.0
    assert BlenderClient._advertised_idle(12.5) == 12.5


@pytest.mark.asyncio
async def test_idle_connection_is_replaced_before_a_call_is_sent(mock_server):
    """Close to the add-on's idle timeout, the call goes out once, on a fresh connection."""
    mock_server.hello_overrides = {"idle_timeout": 300}
    client = BlenderClient(mock_server.host, mock_server.port)
    await client.connect()
    try:
        assert client._idle_timeout == 300.0
        client._last_io -= 300 - blender_client.IDLE_MARGIN
        result = await client.send_command("ping", timeout=5.0)
        assert result["pong"] is True
        assert len(mock_server.hellos) == 2
        assert [command["method"] for command in mock_server.commands] == ["ping"]
        assert client._resume_port is None  # used by the reconnect
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_a_connection_short_of_the_idle_timeout_is_kept(mock_server):
    mock_server.hello_overrides = {"idle_timeout": 300}
    client = BlenderClient(mock_server.host, mock_server.port)
    await client.connect()
    try:
        client._last_io -= 100
        await client.send_command("ping", timeout=5.0)
        assert len(mock_server.hellos) == 1
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_an_addon_without_an_idle_timeout_keeps_the_connection(mock_server):
    """An add-on that does not advertise one (0.5.0) is never treated as idle."""
    client = BlenderClient(mock_server.host, mock_server.port)
    await client.connect()
    try:
        assert client._idle_timeout == 0.0
        client._last_io -= 10**6
        await client.send_command("ping", timeout=5.0)
        assert len(mock_server.hellos) == 1
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_a_call_waiting_for_its_reply_is_not_idle(mock_server):
    mock_server.hello_overrides = {"idle_timeout": 300}
    client = BlenderClient(mock_server.host, mock_server.port)
    await client.connect()
    try:
        client._last_io -= 10**6
        assert client._idle_expired()
        client._pending_requests["x"] = ("render", asyncio.get_running_loop().create_future())
        assert not client._idle_expired()
        client._pending_requests.clear()
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_idle_close_by_blender_is_not_a_dropped_call(mock_server):
    """Blender closing an idle connection fails nothing; the next call reconnects."""
    mock_server.hello_overrides = {"idle_timeout": 300}
    client = BlenderClient(mock_server.host, mock_server.port)
    await client.connect()
    try:
        client._last_io -= 400
        await mock_server.drop_clients()
        for _ in range(100):
            if not client.connected:
                break
            await asyncio.sleep(0.01)
        assert not client.connected
        assert client._resume_port == mock_server.port
        result = await client.send_command("ping", timeout=5.0)
        assert result["pong"] is True
        assert len(mock_server.hellos) == 2
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_idle_reconnect_stays_on_the_same_blender(monkeypatch, mock_server):
    """With discovery, an idle reconnect keeps the Blender in use over a newer one."""
    async with MockBlenderServer() as newer:
        older = {"port": mock_server.port, "blender": "4.5", "blend": "", "pid": 1}
        newest = {"port": newer.port, "blender": "4.5", "blend": "", "pid": 2}
        for server in (mock_server, newer):
            server.hello_overrides = {"idle_timeout": 300}
        client = BlenderClient(mock_server.host)
        found = [[older]]

        async def discover():
            return found[0]

        monkeypatch.setattr(client, "_discover", discover)
        await client.connect()
        try:
            assert client.port == mock_server.port
            found[0] = [newest, older]  # a newer Blender started meanwhile
            client._last_io -= 400
            await client.send_command("ping", timeout=5.0)
            assert client.port == mock_server.port
            assert len(newer.hellos) == 0
            # Any other reconnect still picks the newest, as before.
            await mock_server.drop_clients()
            for _ in range(100):
                if not client.connected:
                    break
                await asyncio.sleep(0.01)
            assert client._resume_port is None  # a drop after activity is not idle
            await client.send_command("ping", timeout=5.0)
            assert client.port == newer.port
        finally:
            await client.disconnect()
