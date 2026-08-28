"""
mitmproxy addon: capture draft pick messages from fantasydraft.espn.com

Run with:
    mitmdump -s draft_capture.py --allow-hosts 'fantasydraft.espn.com -q'

Expected WebSocket message format (plain text frame):
    SELECTED <team_number> <player_id> <round>

Example:
    SELECTED 13 4362238 3
"""

import json
import queue
import re
import sys
import threading
from dataclasses import asdict, dataclass

import zmq
from mitmproxy import http

TARGET_HOST = "fantasydraft.espn.com"

# Set True to see outbound frames, unmatched lines, and connection
# open/close events on stderr. Leave False for clean pick-only output.
DEBUG = False

# ZMQ PUB socket — the mitmproxy process binds and publishes; any number
# of subscribers (your larger app, multiple consumers, a debug tap) can
# connect independently without touching this script.
ZMQ_ENABLED = True
ZMQ_BIND_ADDRESS = "tcp://*:5555"
ZMQ_TOPIC = "draft.pick"

# Path to the id -> player name lookup. Adjust if you keep it elsewhere;
# relative paths are resolved from wherever you run mitmdump.
PLAYERS_FILE = "players.json"

try:
    with open(PLAYERS_FILE, "r", encoding="utf-8") as f:
        PLAYERS = json.load(f)
except FileNotFoundError:
    print(f"[warning] {PLAYERS_FILE} not found — picks will print raw "
          f"player IDs instead of names. Run mitmdump from the directory "
          f"containing {PLAYERS_FILE}, or edit PLAYERS_FILE above.",
          file=sys.stderr)
    PLAYERS = {}


def player_name(player_id: int) -> str:
    return PLAYERS.get(str(player_id), f"Unknown player ({player_id})")

# Matches a single "data: SELECTED ..." line, with an optional trailing
# token (observed as a GUID, present only for the connected client's own
# pick). Applied per-line, not to the whole WS frame.
PICK_PATTERN = re.compile(
    r"SELECTED\s+(\d+)\s+(\d+)\s+(\d+)(?:\s+\S+)?"
)


@dataclass
class DraftPickEvent:
    """Normalized event shape — this is what downstream consumers should see."""
    team_number: int
    player_id: int
    round_number: int
    pick_number: int = 0  # assigned after deduplication, see websocket_message
    source: str = "espn_ws"
    raw: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self))


class DraftCapture:
    def __init__(self):
        # websocket_message runs on mitmproxy's asyncio event loop. Any
        # blocking call made directly in that hook (file I/O, HTTP POST,
        # even a slow print) stalls every other flow on the proxy, which
        # gets worse as more sockets open concurrently. Route events
        # through a queue to a background thread instead, so the hook
        # itself only ever does a fast, non-blocking put().
        self._queue: queue.Queue[DraftPickEvent] = queue.Queue()
        self._worker = threading.Thread(target=self._drain_queue, daemon=True)
        self._worker.start()
        self.on_pick = self._queue.put

        # A player can only be selected once. Reconnects replay recent
        # history, so the same SELECTED line can arrive more than once
        # across different socket connections — deduplicate on player_id
        # before it ever reaches the queue.
        self._seen_player_ids: set[int] = set()

        # Running count of unique picks seen this session. Only accurate
        # if the script was attached before pick 1 of the draft.
        self._pick_count = 0

        self._zmq_socket = None
        if ZMQ_ENABLED:
            try:
                zmq_context = zmq.Context.instance()
                self._zmq_socket = zmq_context.socket(zmq.PUB)
                self._zmq_socket.bind(ZMQ_BIND_ADDRESS)
                print(f"[zmq] publishing on {ZMQ_BIND_ADDRESS}, "
                      f"topic {ZMQ_TOPIC!r}", file=sys.stderr, flush=True)
            except zmq.ZMQError as exc:
                print(f"[zmq] failed to bind {ZMQ_BIND_ADDRESS}: {exc}",
                      file=sys.stderr, flush=True)
                self._zmq_socket = None

    def _drain_queue(self) -> None:
        while True:
            event = self._queue.get()
            try:
                self._sink(event)
            except Exception as exc:  # noqa: BLE001
                # A failing sink (server down, disk full, etc.) must not
                # kill the worker thread or back up the queue silently.
                print(f"[sink error] {event.to_json()}: {exc}", file=sys.stderr, flush=True)

    def _sink(self, event: DraftPickEvent) -> None:
        print(f"{event.pick_number}. {player_name(event.player_id)}", flush=True)

        if self._zmq_socket is not None:
            # Multipart: [topic, payload]. Subscribers filter by topic
            # prefix, so you can add other topics later (e.g. "draft.clock",
            # "draft.on_the_clock") without subscribers needing to filter
            # unrelated messages themselves.
            self._zmq_socket.send_multipart([
                ZMQ_TOPIC.encode("utf-8"),
                event.to_json().encode("utf-8"),
            ])

        # Full event detail, if you need team/round/raw for debugging:
        # print(event.to_json(), flush=True)
        #
        # Other sink options (pick any, or combine with zmq above):
        #
        # 1) Append to a local file the main app tails:
        # with open("picks.jsonl", "a") as f:
        #     f.write(event.to_json() + "\n")
        #
        # 2) POST to a local server your tracker app runs:
        # import urllib.request
        # req = urllib.request.Request(
        #     "http://127.0.0.1:8765/pick",
        #     data=event.to_json().encode(),
        #     headers={"Content-Type": "application/json"},
        # )
        # urllib.request.urlopen(req, timeout=1)

    def parse_message(self, text: str) -> list[DraftPickEvent]:
        """A single WS frame can contain multiple 'data: X' entries
        (CLOCK, AUTOSUGGEST, JOINED, PONG, SELECTING, SELECTED, ...),
        separated by blank lines. Scan every line and pull out every
        SELECTED event found, rather than assuming the whole frame is
        one message."""
        events = []
        for line in text.splitlines():
            match = PICK_PATTERN.search(line)
            if not match:
                continue
            team_number, player_id, round_number = match.groups()
            events.append(DraftPickEvent(
                team_number=int(team_number),
                player_id=int(player_id),
                round_number=int(round_number),
                raw=line.strip(),
            ))
        return events

    def websocket_message(self, flow: http.HTTPFlow):
        if TARGET_HOST not in flow.request.pretty_host:
            return

        if not flow.websocket or not flow.websocket.messages:
            # websocket_start can fire before any frame has arrived;
            # nothing to read yet on this event.
            return
        message = flow.websocket.messages[-1]

        # Only handle text frames; binary frames (ping/pong, protobuf, etc.)
        # are skipped rather than raising.
        if message.is_text:
            try:
                text = message.content if isinstance(message.content, str) \
                    else message.content.decode("utf-8")
            except UnicodeDecodeError:
                return
        else:
            return

        # from_client distinguishes outbound (your own actions) from
        # inbound (server broadcasts). Broadcasts typically aren't
        # echoed back to the sender, so your own picks likely never
        # appear as a received SELECTED line — they'd show up here,
        # in whatever format the client sends when you make a pick.
        if message.from_client:
            if DEBUG:
                print(f"[outbound] {text!r}", file=sys.stderr, flush=True)
            return

        try:
            events = self.parse_message(text)
        except Exception as exc:  # noqa: BLE001
            print(f"[parse error] {text!r}: {exc}", file=sys.stderr, flush=True)
            return

        for event in events:
            if event.player_id in self._seen_player_ids:
                # Replay from a reconnect — already emitted this pick.
                continue
            self._seen_player_ids.add(event.player_id)
            self._pick_count += 1
            event.pick_number = self._pick_count
            self.on_pick(event)

    def websocket_start(self, flow: http.HTTPFlow):
        if DEBUG and TARGET_HOST in flow.request.pretty_host:
            print(f"[ws open] {flow.request.pretty_url}", file=sys.stderr, flush=True)

    def websocket_end(self, flow: http.HTTPFlow):
        if DEBUG and TARGET_HOST in flow.request.pretty_host:
            print(f"[ws close] {flow.request.pretty_url}", file=sys.stderr, flush=True)


addons = [DraftCapture()]