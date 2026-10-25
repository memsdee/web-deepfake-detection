/* ==========================================================================
 HỢP ĐỒNG API & CẤU HÌNH (app.py)
 ========================================================================== */
export const API_BASE = "";                      // "" = cùng origin; đổi sang URL tunnel nếu cần
export const UPLOAD_FIELD = "file";              // FastAPI: file: UploadFile = File(...)
export const ENDPOINT_SUBMIT = "/api/analyze";   // POST multipart -> {"job_id": "..."}
export const ENDPOINT_JOB = "/api/jobs/";        // GET -> {"status","result"|"error",...}
export const HEALTH_ENDPOINT = "/api/health";    // GET -> {"status":"ok","workers",...}
export const HEALTH_POLL_MS = 15000;             // chu kỳ hỏi trạng thái máy chủ
export const STATUS = { QUEUED: "queued", RUNNING: "running", DONE: "done", ERROR: "error" };

/* Ngưỡng & giới hạn (khớp UI: tối đa 15 giây, 60 MB) */
export const DEFAULT_LOW = 0.35;
export const DEFAULT_HIGH = 0.55;
export const MAX_MB = 60;
export const MAX_SECONDS = 15;
export const POLL_MS = 2000;
export const CLIENT_TIMEOUT_MS = 5 * 60 * 1000;
export const RESULT_TTL_MIN = 10;
