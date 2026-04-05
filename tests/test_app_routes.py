import base64
import json

from fastapi.testclient import TestClient

from lottie_env.server.app import app


class TestWebSocketLifecycle:
    def test_reset_returns_base64_frames(self):
        client = TestClient(app)

        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"type": "reset", "data": {}}))
            reset_resp = json.loads(ws.receive_text())
            assert reset_resp["type"] == "observation"
            obs = reset_resp["data"]["observation"]
            assert reset_resp["data"]["reward"] == 0.0

            for field in ["start_frame", "middle_frame", "end_frame"]:
                raw = base64.b64decode(obs[field])
                assert raw[:8] == b"\x89PNG\r\n\x1a\n"

    def test_full_ws_lifecycle(self, bouncing_ball_json: str):
        client = TestClient(app)

        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"type": "reset", "data": {}}))
            reset_resp = json.loads(ws.receive_text())
            assert reset_resp["type"] == "observation"
            obs = reset_resp["data"]["observation"]
            assert reset_resp["data"]["reward"] == 0.0

            for field in ["start_frame", "middle_frame", "end_frame"]:
                assert len(base64.b64decode(obs[field])) > 0

            ws.send_text(
                json.dumps(
                    {"type": "step", "data": {"lottie_json": bouncing_ball_json}}
                )
            )
            step_resp = json.loads(ws.receive_text())
            assert step_resp["type"] == "observation"
            step_obs = step_resp["data"]["observation"]
            assert step_resp["data"]["reward"] >= 0.95

            for field in [
                "submitted_start_frame",
                "submitted_middle_frame",
                "submitted_end_frame",
            ]:
                raw = base64.b64decode(step_obs[field])
                assert raw[:8] == b"\x89PNG\r\n\x1a\n"

            ws.send_text(json.dumps({"type": "step", "data": {"lottie_json": "bad"}}))
            bad_resp = json.loads(ws.receive_text())
            assert bad_resp["data"]["reward"] == -1.0
            assert bad_resp["data"]["observation"]["submitted_start_frame"] == ""

            ws.send_text(json.dumps({"type": "close", "data": {}}))
