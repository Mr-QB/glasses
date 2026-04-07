from __future__ import annotations

from queue import Empty, Queue
from threading import Event, Thread


class SpeechSynthesizer:
    def __init__(
        self,
        rate: int = 165,
        volume: float = 1.0,
        fallback_to_console: bool = True,
    ) -> None:
        self.rate = rate
        self.volume = volume
        self.fallback_to_console = fallback_to_console
        self._messages: Queue[str] = Queue(maxsize=20)
        self._stop_event = Event()
        self._thread: Thread | None = None

        self._engine = None
        self._engine_error: str | None = None
        try:
            import pyttsx3  # type: ignore

            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)
            self._engine = engine
        except Exception as exc:
            self._engine_error = str(exc)

    @property
    def available(self) -> bool:
        return self._engine is not None

    @property
    def engine_error(self) -> str | None:
        return self._engine_error

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._thread = None

    def speak(self, message: str) -> None:
        text = message.strip()
        if not text:
            return

        if self._messages.full():
            try:
                self._messages.get_nowait()
            except Empty:
                pass

        self._messages.put_nowait(text)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                message = self._messages.get(timeout=0.2)
            except Empty:
                continue

            if self._engine is not None:
                try:
                    self._engine.say(message)
                    self._engine.runAndWait()
                    continue
                except Exception as exc:
                    self._engine_error = str(exc)
                    self._engine = None

            if self.fallback_to_console:
                print(f"[VOICE] {message}")
