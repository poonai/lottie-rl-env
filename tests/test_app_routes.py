import json

from fastapi.testclient import TestClient

from lottie_env.server.app import app


class TestFrameRoutes:
    def test_get_valid_frames(self):
        client = TestClient(app)
        for name in ["frame_start", "frame_middle", "frame_end"]:
            resp = client.get(f"/frames/bouncing_ball/{name}")
            assert resp.status_code == 200
            assert resp.headers["content-type"] == "image/png"

    def test_invalid_frame_name_404(self):
        client = TestClient(app)
        resp = client.get("/frames/bouncing_ball/bad_name")
        assert resp.status_code == 404

    def test_nonexistent_task_404(self):
        client = TestClient(app)
        resp = client.get("/frames/no_such_task/frame_start")
        assert resp.status_code == 404


class TestSubmissionRoutes:
    def test_invalid_submitted_frame_name_404(self):
        client = TestClient(app)
        resp = client.get("/submissions/fake_ep/step_1/bad_name")
        assert resp.status_code == 404

    def test_nonexistent_submission_404(self):
        client = TestClient(app)
        resp = client.get("/submissions/no_ep/step_1/frame_start")
        assert resp.status_code == 404


class TestWebSocketLifecycle:
    def test_full_ws_lifecycle(self, bouncing_ball_json: str):
        client = TestClient(app)

        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"type": "reset", "data": {}}))
            reset_resp = json.loads(ws.receive_text())
            assert reset_resp["type"] == "observation"
            obs = reset_resp["data"]["observation"]
            assert obs["start_frame"].startswith("/frames/")
            assert reset_resp["data"]["reward"] == 0.0

            ws.send_text(
                json.dumps(
                    {"type": "step", "data": {"lottie_json": bouncing_ball_json}}
                )
            )
            step_resp = json.loads(ws.receive_text())
            assert step_resp["type"] == "observation"
            step_obs = step_resp["data"]["observation"]
            assert step_resp["data"]["reward"] >= 0.95
            assert "/submissions/" in step_obs["submitted_start_frame"]

            subbmitted_url = step_obs["submitted_start_frame"]
            http_resp = client.get(subbmitted_url)
            assert http_resp.status_code == 200
            assert http_resp.headers["content-type"] == "image/png"

            ws.send_text(json.dumps({"type": "step", "data": {"lottie_json": "bad"}}))
            bad_resp = json.loads(ws.receive_text())
            assert bad_resp["data"]["reward"] == -1.0
            assert bad_resp["data"]["observation"]["submitted_start_frame"] == ""

            ws.send_text(json.dumps({"type": "close", "data": {}}))
