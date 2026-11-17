"""
Asks for, checks and remembers the PostgreSQL login used at startup.

Resolution order (resolve_login):
    1. Credentials given on the command line: used if they work, never saved.
    2. The saved login: used silently if it still works.
    3. Otherwise a login prompt, a window where there's a display and the
       console where there isn't, repeated until a login works or the user
       closes it.

The saved login lives in the user's profile, not beside the exe, so it
survives reinstalls. On Windows the password is encrypted with DPAPI, which
only the same Windows user can decrypt; elsewhere the file is readable by the
owner only.
"""

import base64
import getpass
import json
import os
import sys
import threading
from dataclasses import dataclass

from loguru import logger

from amf.helper.db_config import (
    DEFAULT_PASSWORD,
    DEFAULT_PORT,
    DEFAULT_USER,
    check_login,
)

if sys.platform == "win32":
    _login_dir = os.path.join(
        os.environ.get("APPDATA") or os.path.expanduser("~"), "MycorrhizaFinder"
    )
else:
    _login_dir = os.path.join(
        os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config"),
        "mycorrhizafinder",
    )
login_path = os.path.join(_login_dir, "db_login.json")

if getattr(sys, "frozen", False):  # i.e. a pyinstaller build
    _icon_path = os.path.join(
        sys._MEIPASS,  # type: ignore[attr-defined]
        "_internal",
        "static",
        "amfbrowser.ico",
    )
else:  # i.e. dev
    _icon_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "_internal",
        "static",
        "amfbrowser.ico",
    )

# Matches PrimaryButton.jsx in the frontend.
_TEAL = "#008285"
_TEAL_HOVER = "#016B70"
_HINT_GREY = "#8a8a8a"
_ERROR_RED = "#c40041"

_DEFAULTS_HINT = (
    f"Leave a field blank to use its default: username {DEFAULT_USER}, "
    f"password {DEFAULT_PASSWORD}, port {DEFAULT_PORT}."
)


@dataclass
class Login:
    user: str
    password: str
    port: int


def resolve_login(
    user: str | None = None,
    password: str | None = None,
    port: int | None = None,
    force_prompt: bool = False,
) -> Login | None:
    """
    Finds a working login, prompting for one if needed.

    :param user: Username from the command line, if any.
    :param password: Password from the command line, if any.
    :param port: Port from the command line, if any.
    :param force_prompt: Show the prompt even if a saved login works.
    :return: A login that check_login accepted, or None if the user closed the
        prompt.
    """
    if user is not None or password is not None or port is not None:
        given = Login(
            user or DEFAULT_USER,
            DEFAULT_PASSWORD if password is None else password,
            port or DEFAULT_PORT,
        )
        error = check_login(given.user, given.password, given.port)
        if error is None:
            _log_login(given, "from the command line")
            return given
        logger.warning(f"The login given on the command line didn't work: {error}")
        return _prompt_and_save(Login(given.user, "", given.port), error)

    saved = load_login()
    if saved is not None and not force_prompt:
        error = check_login(saved.user, saved.password, saved.port)
        if error is None:
            _log_login(saved, "saved")
            return saved
        logger.warning(f"The saved PostgreSQL login no longer works: {error}")
        return _prompt_and_save(
            Login(saved.user, "", saved.port),
            f"Your saved login no longer works: {error}",
        )

    return _prompt_and_save(
        None if saved is None else Login(saved.user, "", saved.port), None
    )


def _prompt_and_save(prefill: Login | None, message: str | None) -> Login | None:
    result = _prompt(prefill, message)
    if result is None:
        return None
    login, remember = result
    _log_login(login, "entered")
    if remember:
        save_login(login)
    return login


def _log_login(login: Login, source: str) -> None:
    # Never log the password.
    logger.info(
        f"Logged in to PostgreSQL as '{login.user}' on port {login.port} ({source})."
    )


############# STORAGE ################


def save_login(login: Login) -> None:
    if sys.platform == "win32":
        password = base64.b64encode(_dpapi(login.password.encode(), protect=True))
        record = {
            "user": login.user,
            "port": login.port,
            "password": password.decode("ascii"),
            "protection": "dpapi",
        }
    else:
        record = {
            "user": login.user,
            "port": login.port,
            "password": login.password,
            "protection": "none",
        }

    try:
        os.makedirs(_login_dir, exist_ok=True)
        # Create owner-only from the start rather than chmod-ing afterwards, so
        # the password is never briefly readable by others. Windows ignores the
        # mode; DPAPI covers it there.
        descriptor = os.open(login_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(record, file)
        logger.info(f"Saved the PostgreSQL login to {login_path}.")
    except OSError as error:
        logger.warning(f"Couldn't save the PostgreSQL login to {login_path}: {error}")


def load_login() -> Login | None:
    if not os.path.exists(login_path):
        return None
    try:
        with open(login_path, encoding="utf-8") as file:
            record = json.load(file)
        if record.get("protection") == "dpapi":
            if sys.platform != "win32":
                raise ValueError("it was encrypted on Windows")
            password = _dpapi(base64.b64decode(record["password"]), protect=False)
            record["password"] = password.decode()
        return Login(str(record["user"]), str(record["password"]), int(record["port"]))
    except (OSError, ValueError, KeyError, TypeError) as error:
        # e.g. hand-edited, or copied from another Windows account, whose
        # DPAPI key this one doesn't have.
        logger.warning(f"Ignoring the saved PostgreSQL login in {login_path}: {error}")
        return None


def forget_login() -> None:
    if os.path.exists(login_path):
        os.remove(login_path)
        logger.info(f"Removed the saved PostgreSQL login from {login_path}.")


def _dpapi(data: bytes, protect: bool) -> bytes:
    """Encrypts or decrypts ``data`` with the current Windows user's DPAPI key."""
    import ctypes
    from ctypes import wintypes

    class DataBlob(ctypes.Structure):
        _fields_ = [
            ("cbData", wintypes.DWORD),
            ("pbData", ctypes.POINTER(ctypes.c_char)),
        ]

    buffer = ctypes.create_string_buffer(data, len(data))
    blob_in = DataBlob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char)))
    blob_out = DataBlob()
    crypt32 = ctypes.windll.crypt32  # type: ignore[attr-defined]
    function = crypt32.CryptProtectData if protect else crypt32.CryptUnprotectData
    # CRYPTPROTECT_UI_FORBIDDEN: fail rather than show a Windows dialog.
    if not function(
        ctypes.byref(blob_in), None, None, None, None, 0x1, ctypes.byref(blob_out)
    ):
        raise ValueError(f"Windows couldn't {'en' if protect else 'de'}crypt it")
    try:
        return ctypes.string_at(blob_out.pbData, blob_out.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(blob_out.pbData)  # type: ignore[attr-defined]


############# PROMPT ################


def _prompt(prefill: Login | None, message: str | None) -> tuple[Login, bool] | None:
    """
    Asks for a login until one passes check_login.

    :return: The login and whether to remember it, or None if the user closed
        the prompt.
    """
    try:
        import tkinter
    except ImportError:
        logger.debug("tkinter isn't available, asking for the login in the console.")
        return _prompt_console(prefill, message)
    try:
        return _prompt_window(prefill, message)
    except tkinter.TclError as error:  # e.g. no display to open the window on
        logger.debug(f"Login window unavailable ({error}), asking in the console.")
        return _prompt_console(prefill, message)


def _parse_port(text: str) -> int | None:
    text = text.strip()
    if not text:
        return DEFAULT_PORT
    if text.isdigit() and 1 <= int(text) <= 65535:
        return int(text)
    return None


def _prompt_window(
    prefill: Login | None, message: str | None
) -> tuple[Login, bool] | None:
    # Imported here so a Python without tkinter still reaches the console prompt.
    import tkinter as tk
    from tkinter import font as tkfont

    if sys.platform == "win32":
        try:  # Crisp text on scaled displays instead of a blurry bitmap.
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)  # type: ignore[attr-defined]
        except Exception:
            pass

    root = tk.Tk()
    root.withdraw()
    root.title("MycorrhizaFinder – PostgreSQL login")
    root.resizable(False, False)
    try:
        root.iconbitmap(_icon_path)
    except tk.TclError:
        pass
    for name in ("TkDefaultFont", "TkTextFont"):
        tkfont.nametofont(name).configure(size=10)

    class HintEntry(tk.Entry):
        """
        An entry showing its default in grey whenever it's empty, focused or
        not, so the default stays visible until the user starts typing.
        """

        def __init__(self, master: tk.Misc, hint: str, secret: bool = False) -> None:
            super().__init__(master, width=30)
            self.hint, self.secret = hint, secret
            self.text_colour = self.cget("fg")
            self.showing_hint = False
            # Widget bindings run before the Entry class's own, so the hint is
            # cleared just before the first typed or pasted text goes in.
            self.bind("<KeyPress>", self.on_key)
            self.bind("<<Paste>>", self.hide_hint)
            self.bind("<KeyRelease>", self.show_hint)  # emptied again
            self.bind("<FocusOut>", self.show_hint)
            for event in ("<FocusIn>", "<ButtonRelease-1>"):
                self.bind(event, self.park_cursor)
            self.show_hint()

        def on_key(self, event: "tk.Event[tk.Entry]") -> str | None:
            if not self.showing_hint:
                return None
            if event.char and event.char.isprintable():
                self.hide_hint()
            elif event.keysym in ("BackSpace", "Delete"):
                return "break"  # would otherwise delete part of the hint
            return None

        def park_cursor(self, _event: object = None) -> None:
            if self.showing_hint:
                self.after_idle(lambda: (self.icursor(0), self.selection_clear()))

        def show_hint(self, _event: object = None) -> None:
            if not self.showing_hint and not self.get():
                self.showing_hint = True
                self.configure(show="", fg=_HINT_GREY)
                self.insert(0, self.hint)
                self.icursor(0)

        def hide_hint(self, _event: object = None) -> None:
            if self.showing_hint:
                self.showing_hint = False
                self.delete(0, "end")
                self.configure(show="•" if self.secret else "", fg=self.text_colour)

        def value(self) -> str:
            return "" if self.showing_hint else self.get()

        def set_value(self, text: str) -> None:
            self.hide_hint()
            self.delete(0, "end")
            self.insert(0, text)
            self.show_hint()

    frame = tk.Frame(root, padx=24, pady=20)
    frame.pack(fill="both", expand=True)

    tk.Label(
        frame,
        text=(
            "MycorrhizaFinder keeps its data in a PostgreSQL database called "
            "'amf'. Log in with the PostgreSQL account on this computer."
        ),
        wraplength=380,
        justify="left",
    ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 14))

    user_entry = HintEntry(frame, DEFAULT_USER)
    password_entry = HintEntry(frame, DEFAULT_PASSWORD, secret=True)
    port_entry = HintEntry(frame, str(DEFAULT_PORT))
    for row, (label, entry) in enumerate(
        (("Username", user_entry), ("Password", password_entry), ("Port", port_entry)),
        start=1,
    ):
        tk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=4)
        entry.grid(row=row, column=1, sticky="ew", padx=(12, 0), pady=4, ipady=3)
    frame.columnconfigure(1, weight=1)

    tk.Label(
        frame, text=_DEFAULTS_HINT, fg=_HINT_GREY, wraplength=380, justify="left"
    ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(4, 8))

    remember = tk.BooleanVar(value=True)
    tk.Checkbutton(frame, text="Remember me on this computer", variable=remember).grid(
        row=5, column=0, columnspan=2, sticky="w"
    )

    status = tk.Label(
        frame, text=message or "", fg=_ERROR_RED, wraplength=380, justify="left"
    )
    status.grid(row=6, column=0, columnspan=2, sticky="w", pady=(10, 0))

    buttons = tk.Frame(frame)
    buttons.grid(row=7, column=0, columnspan=2, sticky="e", pady=(14, 0))
    cancel_button = tk.Button(buttons, text="Cancel", width=10, command=root.destroy)
    cancel_button.pack(side="right")
    login_button = tk.Button(
        buttons,
        text="Log in",
        width=10,
        bg=_TEAL,
        fg="white",
        activebackground=_TEAL_HOVER,
        activeforeground="white",
        relief="flat",
        cursor="hand2",
    )
    login_button.pack(side="right", padx=(0, 8))

    if prefill is not None:
        user_entry.set_value(prefill.user if prefill.user != DEFAULT_USER else "")
        port_entry.set_value(str(prefill.port) if prefill.port != DEFAULT_PORT else "")

    outcome: list[tuple[Login, bool]] = []
    checking = threading.Event()

    def show_status(text: str, colour: str = _ERROR_RED) -> None:
        status.configure(text=text, fg=colour)

    def submit(_event: object = None) -> None:
        if checking.is_set():
            return
        port = _parse_port(port_entry.value())
        if port is None:
            show_status("Port must be a whole number between 1 and 65535.")
            port_entry.focus_set()
            return
        login = Login(
            user_entry.value().strip() or DEFAULT_USER,
            password_entry.value() or DEFAULT_PASSWORD,
            port,
        )

        # check_login can block for up to its connect timeout, so it runs off
        # the UI thread and is polled for, keeping the window responsive.
        checking.set()
        login_button.configure(state="disabled", cursor="")
        show_status("Connecting…", _HINT_GREY)
        result: dict[str, str | None] = {}
        worker = threading.Thread(
            target=lambda: result.update(
                error=check_login(login.user, login.password, login.port)
            ),
            daemon=True,
        )
        worker.start()

        def poll() -> None:
            if worker.is_alive():
                root.after(100, poll)
                return
            checking.clear()
            login_button.configure(state="normal", cursor="hand2")
            if result["error"] is None:
                outcome.append((login, remember.get()))
                root.destroy()
                return
            show_status(result["error"])
            password_entry.set_value("")
            password_entry.focus_set()

        root.after(100, poll)

    login_button.configure(command=submit)
    root.bind("<Return>", submit)
    root.bind("<Escape>", lambda _event: root.destroy())
    root.protocol("WM_DELETE_WINDOW", root.destroy)

    # Centre it and bring it in front of the console window that launched it.
    root.update_idletasks()
    x = (root.winfo_screenwidth() - root.winfo_reqwidth()) // 2
    y = (root.winfo_screenheight() - root.winfo_reqheight()) // 3
    root.geometry(f"+{x}+{y}")
    root.deiconify()
    root.attributes("-topmost", True)
    root.after(500, lambda: root.attributes("-topmost", False))
    root.focus_force()
    (password_entry if prefill is not None else user_entry).focus_set()

    root.mainloop()
    return outcome[0] if outcome else None


def _prompt_console(
    prefill: Login | None, message: str | None
) -> tuple[Login, bool] | None:
    default_user = prefill.user if prefill is not None else DEFAULT_USER
    default_port = prefill.port if prefill is not None else DEFAULT_PORT

    if sys.stdin is None or not sys.stdin.isatty():
        # Nobody to ask (e.g. a scheduled job), so try the defaults once, as
        # the tool did before it had a prompt.
        login = Login(default_user, DEFAULT_PASSWORD, default_port)
        error = check_login(login.user, login.password, login.port)
        if error is None:
            return login, False
        logger.error(
            f"Couldn't log in to PostgreSQL with the defaults, and there's no window "
            f"or terminal to ask for a login: {error} Log in once interactively, or "
            f"pass --db-user, --db-password and --db-port."
        )
        # ERR_NO_DATABASE_CONNECTION = 42, as in db_config.connect
        sys.exit(42)

    print("\nMycorrhizaFinder - PostgreSQL login")
    print(_DEFAULTS_HINT + " Press Ctrl+C to cancel.")
    if message:
        print(message)
    while True:
        try:
            user = input(f"Username [{default_user}]: ").strip() or default_user
            password = getpass.getpass(f"Password [{DEFAULT_PASSWORD}]: ")
            port = _parse_port(input(f"Port [{default_port}]: ") or str(default_port))
            if port is None:
                print("Port must be a whole number between 1 and 65535.")
                continue
            login = Login(user, password or DEFAULT_PASSWORD, port)
            print("Connecting...")
            error = check_login(login.user, login.password, login.port)
            if error is not None:
                print(error)
                continue
            answer = input("Remember this login on this computer? [Y/n]: ")
            return login, answer.strip().lower() not in ("n", "no")
        except (EOFError, KeyboardInterrupt):
            print()
            return None
