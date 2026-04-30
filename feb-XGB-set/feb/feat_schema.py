# feature 구조. 순서 및 내용 변경  X
# feature의 이름과 순서를 고정하는 파일
FEATURE_SCHEMA = {
    # G1 : length feature
    "url_length": int,                 # 전체 URL 길이
    "hostname_length": int,           # 도메인 길이
    "path_length": int,               # 경로(path) 길이
    "query_length": int,              # 쿼리 문자열 길이
    "subdomain_count": int,           # 서브도메인 개수

    # G2 : character feature
    "dot_count": int,                 # '.' 개수
    "hyphen_count": int,              # '-' 개수
    "underscore_count": int,          # '_' 개수
    "slash_count": int,               # '/' 개수
    "question_mark_count": int,       # '?' 개수
    "equal_count": int,               # '=' 개수
    "ampersand_count": int,           # '&' 개수
    "at_count": int,                  # '@' 개수 (우회 공격에 자주 사용)
    "percent_count": int,             # '%' 개수 (인코딩 흔적)
    "special_char_count": int,        # 특수문자 총 개수

    # G3 : keyword feature
    "keyword_count": int,             # 민감 키워드 총 등장 개수
    "has_sensitive_keyword": int,     # 민감 키워드 존재 여부
    "has_free_hosting_keyword": int,  # 무료 호스팅 도메인 포함 여부
    "has_suspicious_extension": int,  # 실행파일/압축파일 확장자 존재 여부

    # G4 : numeric feature
    "digit_count": int,               # 숫자 개수
    "digit_ratio": float,             # 숫자 비율 (숫자/전체 길이)
    "has_ip_address": int,            # IP 주소 형태 URL 여부

    # G5 : structure/behavioral feature
    "is_https": int,                  # HTTPS 사용 여부
    "is_shortened_url": int,          # 단축 URL 여부
    "redirect_count": int,            # 리다이렉트 횟수
    "has_redirect_loop": int,         # 리다이렉트 루프 여부
    "rule_flag_count": int,           # 룰 기반 탐지 개수
    "suspicious_param_count": int,    # 의심 파라미터 개수
    "embedded_url_count": int,        # URL 내부에 포함된 URL 개수

    # G6 (optional) : external/verification feature
    "certificate_valid": int,         # SSL 인증서 유효 여부
    "safe_browsing_threat": int,      # 외부 API에서 악성 판정 여부
}

# 모델 입력 순서 고정용
FEATURE_COLUMNS = list(FEATURE_SCHEMA.keys())