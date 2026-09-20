"""
Утилиты для работы с окном игры
"""

import win32gui
import win32process
import psutil

from config import PROCESS_NAME


def find_window_by_process(process_name: str = PROCESS_NAME):
    """Поиск окна по имени процесса"""
    target_pids = set()
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if proc.info['name'] and proc.info['name'].lower() == process_name.lower():
                target_pids.add(proc.info['pid'])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    if not target_pids:
        return None

    found = []
    def enum_cb(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        if pid in target_pids and win32gui.GetWindowText(hwnd):
            found.append(hwnd)
    
    win32gui.EnumWindows(enum_cb, None)
    return found[0] if found else None


def get_client_rect(hwnd):
    """Получение координат клиентской области окна"""
    cl, ct, cr, cb = win32gui.GetClientRect(hwnd)
    left, top = win32gui.ClientToScreen(hwnd, (cl, ct))
    right, bottom = win32gui.ClientToScreen(hwnd, (cr, cb))
    return left, top, right - left, bottom - top


def focus_window(hwnd):
    """Установить фокус на окно"""
    try:
        win32gui.SetForegroundWindow(hwnd)
        return True
    except Exception:
        return False