"""Run a Python file (or inline code) inside MY TouchDesigner (port 9995) via the MCP web server.
usage: python tdrun.py file.py | python tdrun.py -c "code"
"""
import sys, json, urllib.request, os

PORT = int(os.environ.get("WAYMO_TD_PORT", "9995"))
URL = "http://127.0.0.1:%d/api/td/server/exec" % PORT


def run(code, timeout=300):
    req = urllib.request.Request(URL, data=json.dumps({"script": code}).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main(argv):
    code = argv[1] if argv[0] == "-c" else open(argv[0], encoding="utf-8").read()
    out = run(code)
    d = out.get("data") or {}
    if d.get("stdout"):
        print(d["stdout"], end="")
    if d.get("stderr"):
        print("STDERR:", d["stderr"], end="")
    if out.get("error"):
        print("ERROR:", out["error"])
    if d.get("result") is not None:
        print("RESULT:", json.dumps(d["result"], indent=1, default=str)[:20000])


if __name__ == "__main__":
    main(sys.argv[1:])
