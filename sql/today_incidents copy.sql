SELECT
    i.batch_name AS 배치명,
    i.status AS 상태,
    i.severity AS 심각도,
    i.error_code AS 오류코드,
    i.error_message AS 오류메시지,
    i.start_time AS 오류발생시간,
    i.end_time AS 종료시간,
    TIMESTAMPDIFF(
        MINUTE,
        i.start_time,
        COALESCE(i.end_time, NOW())
    ) AS 경과분,
    i.customer_impact_yn AS 고객영향여부,
    i.downstream_job_count AS 후행배치수,
    a.action_detail AS 조치내용,
    a.action_owner AS 담당자
FROM {schema}.TB_BATCH_INCIDENT i
LEFT JOIN (
    SELECT *
    FROM (
        SELECT
            h.*,
            ROW_NUMBER() OVER (
                PARTITION BY h.batch_name, h.error_code
                ORDER BY h.created_at DESC
            ) AS rn
        FROM {schema}.TB_BATCH_ACTION_HISTORY h
    ) t
    WHERE rn = 1
) a
    ON i.batch_name = a.batch_name
   AND i.error_code = a.error_code
WHERE DATE(i.start_time) = CURRENT_DATE
  AND i.status = :status
ORDER BY i.start_time DESC
LIMIT :limit_count;