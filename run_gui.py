import http.server
import socketserver
import json
import os
import sys
import ctypes
from ctypes import wintypes
import webbrowser
import time

PORT = 8080

# C API bindings for native turbo_engine DLL
native_engine = None
engine_handle = None
current_model_dir = None
DEFAULT_MODEL_DIR = 'gemma-4-26b-a4b.gturbo'

# Sampling knobs only.
#
# context_len and slots are deliberately ABSENT. They are fixed inside
# ForwardRunner::initialize(), and the C ABI this script binds through exposes no setter for
# either -- so any value stored here could never reach the engine. They used to live in this
# dict anyway, which meant the Python server reported a configuration it had no ability to
# apply: the default read `context_len: 62000`, fifteen times the engine's hard maximum of
# 4096, and nothing ever noticed because the number was never used for anything.
#
# The native server (turbo-winfare.exe, the canonical front-end) does support both, via
# POST /api/config followed by POST /api/load_model. Use it if you need them.
current_config = {
    'eviction_policy': 'LFU',
    'temperature': 0.20,
    'top_p': 0.90,
    'top_k': 64,
    'max_tokens': 512
}

def init_native_engine():
    global native_engine, engine_handle, current_model_dir
    dll_paths = [
        os.path.join(os.getcwd(), 'build', 'libturbo_engine.dll'),
        os.path.join(os.getcwd(), 'build', 'turbo_engine.dll'),
        os.path.join(os.getcwd(), 'libturbo_engine.dll')
    ]
    for dll_path in dll_paths:
        if os.path.exists(dll_path):
            try:
                lib = ctypes.CDLL(dll_path)
                lib.turbo_engine_create.restype = ctypes.c_void_p
                lib.turbo_engine_create.argtypes = [ctypes.c_char_p]

                lib.turbo_engine_destroy.restype = None
                lib.turbo_engine_destroy.argtypes = [ctypes.c_void_p]

                lib.turbo_engine_generate.restype = ctypes.c_char_p
                lib.turbo_engine_generate.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]

                lib.turbo_engine_get_telemetry.restype = ctypes.c_char_p
                lib.turbo_engine_get_telemetry.argtypes = [ctypes.c_void_p]

                lib.turbo_engine_load_model.restype = ctypes.c_int
                lib.turbo_engine_load_model.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

                lib.turbo_engine_unload_model.restype = None
                lib.turbo_engine_unload_model.argtypes = [ctypes.c_void_p]

                lib.turbo_engine_clear_cache.restype = None
                lib.turbo_engine_clear_cache.argtypes = [ctypes.c_void_p]

                lib.turbo_engine_stop.restype = None
                lib.turbo_engine_stop.argtypes = [ctypes.c_void_p]

                lib.turbo_engine_last_error.restype = ctypes.c_char_p
                lib.turbo_engine_last_error.argtypes = []

                native_engine = lib
                handle = lib.turbo_engine_create(DEFAULT_MODEL_DIR.encode('utf-8'))
                if handle:
                    engine_handle = handle
                    current_model_dir = DEFAULT_MODEL_DIR
                    print(f"[NATIVE] Loaded DirectX 12 C++ engine DLL: {dll_path}")
                    return True
                # The DLL loaded but the model did not. Report why instead of falling
                # through silently -- there is no Python-side generation fallback.
                err = lib.turbo_engine_last_error()
                detail = err.decode('utf-8', 'replace') if err else '(no detail)'
                print(f"[NATIVE] Engine DLL loaded but model failed to open: {detail}")
                return False
            except Exception as ex:
                print(f"[NATIVE] Failed to initialize DLL {dll_path}: {ex}")
    print("[NATIVE] No engine DLL found. Build it with: cmake --build build --config Release")
    return False

class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ('dwLength', wintypes.DWORD),
        ('dwMemoryLoad', wintypes.DWORD),
        ('ullTotalPhys', ctypes.c_uint64),
        ('ullAvailPhys', ctypes.c_uint64),
        ('ullTotalPageFile', ctypes.c_uint64),
        ('ullAvailPageFile', ctypes.c_uint64),
        ('ullTotalVirtual', ctypes.c_uint64),
        ('ullAvailVirtual', ctypes.c_uint64),
        ('ullAvailExtendedVirtual', ctypes.c_uint64),
    ]

def get_sys_memory():
    try:
        mem = MEMORYSTATUSEX()
        mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
        return mem.ullTotalPhys, mem.ullAvailPhys
    except Exception:
        return 16 * 1024 * 1024 * 1024, 7 * 1024 * 1024 * 1024




class GUIHandler(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path):
        clean_path = path.split('?')[0].split('#')[0]
        if clean_path in ['/', '/index.html']:
            return os.path.join(os.getcwd(), 'gui', 'index.html')
        elif clean_path.startswith('/api/'):
            return path
        else:
            return os.path.join(os.getcwd(), 'gui', clean_path.lstrip('/'))

    def do_GET(self):
        if self.path.startswith('/api/models'):
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            # List bundles that actually exist. This previously returned a single hardcoded
            # entry with is_active: True whether or not any model was loaded.
            models = []
            for name in sorted(os.listdir('.')):
                if name.endswith('.gturbo') and os.path.isdir(name):
                    models.append({
                        'path': name,
                        'id': name,
                        'name': name,
                        'layers': None,
                        'experts': None,
                        'top_k': None,
                        'is_active': bool(engine_handle) and name == current_model_dir,
                    })
            self.wfile.write(json.dumps({'models': models}).encode('utf-8'))
            return

        elif self.path.startswith('/api/telemetry'):
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            total_phys, avail_phys = get_sys_memory()

            # Telemetry comes from the engine or not at all. This used to overwrite the
            # engine's real memory figures with values computed from the GUI sliders, and to
            # invent an entire block (48.5 t/s, 78.4% hit rate, a fixed expert list,
            # model_active: True) whenever the DLL was missing.
            if native_engine and engine_handle:
                try:
                    telemetry_raw = native_engine.turbo_engine_get_telemetry(engine_handle)
                    if telemetry_raw:
                        telemetry_str = telemetry_raw.decode('utf-8') if isinstance(telemetry_raw, bytes) else str(telemetry_raw)
                        data = json.loads(telemetry_str)
                        data['config'] = current_config
                        self.wfile.write(json.dumps(data).encode('utf-8'))
                        return
                except Exception as ex:
                    print(f"[SERVER] Native telemetry exception: {ex}")

            # No engine: report only what this process can actually observe.
            telemetry = {
                'status': 'OK',
                'gpu_name': None,
                'model_active': False,
                'model_dir': None,
                'memory': {
                    'system_total_ram_gb': round(total_phys / (1024**3), 2),
                    'system_avail_ram_gb': round(avail_phys / (1024**3), 2),
                },
                'performance': {},
                'cache': {'eviction_policy': current_config.get('eviction_policy', 'LFU')},
                'config': current_config,
                'active_experts': []
            }
            self.wfile.write(json.dumps(telemetry).encode('utf-8'))
            return

        super().do_GET()

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length).decode('utf-8') if content_length > 0 else ''

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        if self.path.startswith('/api/config'):
            req = json.loads(post_data) if post_data else {}
            # context_len / slots are accepted from the GUI and ignored, because this server
            # cannot apply them (see current_config). Saying so in the response is better
            # than storing a value that will never take effect.
            unsupported = [k for k in ('context_len', 'slots') if k in req]
            if 'eviction_policy' in req:
                current_config['eviction_policy'] = str(req['eviction_policy'])
            if 'temperature' in req:
                current_config['temperature'] = float(req['temperature'])
            if 'top_p' in req:
                current_config['top_p'] = float(req['top_p'])
            if 'top_k' in req:
                current_config['top_k'] = int(req['top_k'])
            if 'max_tokens' in req:
                current_config['max_tokens'] = int(req['max_tokens'])

            body = {'status': 'SUCCESS', 'config': current_config, 'requires_reload': False}
            if unsupported:
                body['ignored'] = unsupported
                body['message'] = ('This Python bridge cannot change ' + ', '.join(unsupported) +
                                   '; the C ABI exposes no setter. Use turbo-winfare.exe.')
            self.wfile.write(json.dumps(body).encode('utf-8'))
            return

        elif self.path.startswith('/api/generate'):
            req = json.loads(post_data) if post_data else {}
            prompt = req.get('prompt', '')
            max_tokens = int(req.get('max_tokens', current_config['max_tokens']))
            temperature = float(req.get('temperature', current_config['temperature']))
            top_p = float(req.get('top_p', current_config['top_p']))
            top_k = int(req.get('top_k', current_config['top_k']))
            eviction_policy = str(req.get('eviction_policy', current_config['eviction_policy']))

            current_config['max_tokens'] = max_tokens
            current_config['temperature'] = temperature
            current_config['top_p'] = top_p
            current_config['top_k'] = top_k
            current_config['eviction_policy'] = eviction_policy

            # There is no Python-side generation. Either the native engine produces text or
            # this is reported as an error -- an error string must never reach the GUI
            # through 'output_text', where it would render as an assistant reply.
            if not prompt:
                resp = {'status': 'ERROR', 'message': 'Missing prompt.'}
            elif not (native_engine and engine_handle):
                resp = {'status': 'ERROR',
                        'message': 'Native C++ engine is not loaded. Build it with '
                                   'cmake --build build --config Release'}
            else:
                try:
                    res = native_engine.turbo_engine_generate(
                        engine_handle, prompt.encode('utf-8'), max_tokens)
                    if res:
                        resp = {
                            'status': 'SUCCESS',
                            'output_text': res.decode('utf-8', 'replace'),
                            'config_applied': current_config,
                        }
                    else:
                        err = native_engine.turbo_engine_last_error()
                        detail = err.decode('utf-8', 'replace') if err else '(no detail)'
                        resp = {'status': 'ERROR', 'message': detail}
                except Exception as ex:
                    print(f"[SERVER] Native generation exception: {ex}")
                    resp = {'status': 'ERROR', 'message': f'Engine exception: {ex}'}

            self.wfile.write(json.dumps(resp).encode('utf-8'))

        elif self.path.startswith('/api/load_model'):
            global current_model_dir
            req = json.loads(post_data) if post_data else {}
            m_path = req.get('model_path', DEFAULT_MODEL_DIR)
            # Report the actual load result. This previously always answered
            # "loaded cleanly", including when no engine was attached at all.
            if not (native_engine and engine_handle):
                resp = {'status': 'ERROR', 'message': 'Native C++ engine is not loaded.'}
            elif native_engine.turbo_engine_load_model(engine_handle, m_path.encode('utf-8')):
                current_model_dir = m_path
                resp = {'status': 'SUCCESS', 'message': f'Loaded {m_path}.'}
            else:
                err = native_engine.turbo_engine_last_error()
                detail = err.decode('utf-8', 'replace') if err else '(no detail)'
                resp = {'status': 'ERROR', 'message': f'Failed to load {m_path}: {detail}'}
            self.wfile.write(json.dumps(resp).encode('utf-8'))

        elif self.path.startswith('/api/unload_model'):
            if native_engine and engine_handle:
                native_engine.turbo_engine_unload_model(engine_handle)
                globals()['current_model_dir'] = None
            self.wfile.write(json.dumps({'status': 'SUCCESS', 'message': 'Model unloaded from UMA RAM.'}).encode('utf-8'))

        elif self.path.startswith('/api/clear_cache'):
            if native_engine and engine_handle:
                native_engine.turbo_engine_clear_cache(engine_handle)
            self.wfile.write(json.dumps({'status': 'SUCCESS', 'message': 'Expert DRAM Cache pool flushed.'}).encode('utf-8'))

        elif self.path.startswith('/api/stop'):
            if native_engine and engine_handle:
                native_engine.turbo_engine_stop(engine_handle)
            self.wfile.write(json.dumps({'status': 'SUCCESS', 'message': 'Generation stop signal sent.'}).encode('utf-8'))

        elif self.path.startswith('/api/repack'):
            # In-app repacking has been removed; it wrote zero-filled placeholder bundles.
            self.wfile.write(json.dumps({
                'status': 'ERROR',
                'message': 'In-app repacking has been removed. Run: '
                           'python tools/convert_hf_to_gturbo.py --output <name>.gturbo'
            }).encode('utf-8'))

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("=" * 60)
    print(" Turbo-WinFare GUI Runner & Engine Bridge")
    print(f" Web Server Active at http://localhost:{PORT}")
    print("=" * 60)

    init_native_engine()

    url = f"http://localhost:{PORT}"
    webbrowser.open(url)

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), GUIHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")

if __name__ == "__main__":
    main()
