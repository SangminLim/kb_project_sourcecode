# 일반 소득공제 가맹점 파일 생성

## Batch ID
`BATCH_GENERAL_DEDUCT_MERCHANT_EXPORT`

## Batch Type
`db_to_file`

## 설명
[배치 개발 요청서] 배치명: 일반 소득공제 가맹점 파일 생성 기준 테이블: - TB_GENERAL_DEDUCT_MERCHANT 출력 목적: - 일반 소득공제 대상 가맹점 정보 추출 출력 형식: - CSV 파일명: - general_deduct_merchant_YYYYMMDD.csv 기준일자: - APPLY_START_DT 처리 내용: - 일반 소득공제 대상 가맹점 정보를 조회한다. - 기준일자가 적용시작일자와 종료일자 사이인 데이터만 조회한다. - 사용여부가 'Y'인 데이터만 추출한다. 출력 컬럼: - MERCHANT_ID - MERCHANT_NM - BIZ_NO - MCC_CD - APPLY_START_DT - APPLY_END_DT - USE_YN 조건: - USE_YN = 'Y' - 기준일자 BETWEEN APPLY_START_DT AND APPLY_END_DT 배치 유형: - 파일 생성 배치 - 일 배치

## 실행 예시

```bash
python job.py --database-url "$DATABASE_URL" --base-date 20260428 --output-dir ./output
```

## 출력 파일
`general_deduct_merchant_{base_date}.csv`

## 검토 필요사항
- query.sql의 테이블/컬럼/조건이 실제 운영 기준과 맞는지 확인
- 인덱스 사용 여부와 실행 계획 확인
- 파일 구분자, 인코딩, 헤더 포함 여부 확인
- 건수/NULL/중복/금액 합계 검증 조건 추가
