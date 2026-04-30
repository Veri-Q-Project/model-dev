# analyze-engine의 context 읽어오는 용도.
# context에서 FEB feature dict를 만듭니다.
# TODO: context와 형식 비교 후 동기화 필요

import re
from feb.feat_schema import FEATURE_COLUMNS
from feb.feat_keywords import (
    SENSITIVE_KEYWORDS,
    FREE_HOSTING_KEYWORDS,
    SUSPICIOUS_EXTENSIONS,
)


def _safe_get(mapping: dict, key: str, default=None):
    if not isinstance(mapping, dict):
        return default
    return mapping.get(key, default)


def _count_keywords(text: str, keywords: list[str]) -> int:
    lower_text = text.lower()
    return sum(1 for keyword in keywords if keyword in lower_text)


def _has_any(text: str, targets: list[str]) -> int:
    lower_text = text.lower()
    return int(any(target in lower_text for target in targets))


def _is_ip_address(hostname: str) -> int:
    if not hostname:
        return 0

    pattern = r"^\d{1,3}(\.\d{1,3}){3}$"
    return int(bool(re.match(pattern, hostname)))


def extract_features(context) -> dict:
    url_data = getattr(context, "url", {}) or {}
    detection = getattr(context, "detection", {}) or {}
    redirect = getattr(context, "redirect", {}) or {}
    certificate = getattr(context, "certificate", {}) or {}
    external = getattr(context, "external", {}) or {}

    normalized_url = _safe_get(url_data, "normalized_url", "") or ""
    hostname = _safe_get(url_data, "hostname", "") or ""
    path = _safe_get(url_data, "path", "") or ""
    query = _safe_get(url_data, "query", "") or ""

    full_text = normalized_url.lower()

    shortened_url = _safe_get(detection, "shortened_url", {}) or {}
    rule_based = _safe_get(detection, "rule_based", {}) or {}
    safe_browsing = _safe_get(external, "safe_browsing", {}) or {}

    features = {}

    # G1 : length feature
    features["url_length"] = len(normalized_url)
    features["hostname_length"] = len(hostname)
    features["path_length"] = len(path)
    features["query_length"] = len(query)
    features["subdomain_count"] = max(hostname.count(".") - 1, 0) if hostname else 0

    # G2 : character feature
    features["dot_count"] = normalized_url.count(".")
    features["hyphen_count"] = normalized_url.count("-")
    features["underscore_count"] = normalized_url.count("_")
    features["slash_count"] = normalized_url.count("/")
    features["question_mark_count"] = normalized_url.count("?")
    features["equal_count"] = normalized_url.count("=")
    features["ampersand_count"] = normalized_url.count("&")
    features["at_count"] = normalized_url.count("@")
    features["percent_count"] = normalized_url.count("%")
    features["special_char_count"] = len(re.findall(r"[^a-zA-Z0-9]", normalized_url))

    # G3 : keyword feature
    features["keyword_count"] = _count_keywords(full_text, SENSITIVE_KEYWORDS)
    features["has_sensitive_keyword"] = int(features["keyword_count"] > 0)
    features["has_free_hosting_keyword"] = _has_any(full_text, FREE_HOSTING_KEYWORDS)
    features["has_suspicious_extension"] = _has_any(full_text, SUSPICIOUS_EXTENSIONS)

    # G4 : numeric feature
    digit_count = sum(ch.isdigit() for ch in normalized_url)
    features["digit_count"] = digit_count
    features["digit_ratio"] = digit_count / len(normalized_url) if normalized_url else 0.0
    features["has_ip_address"] = _is_ip_address(hostname)

    # G5 : structure/behavioral feature
    features["is_https"] = int(bool(_safe_get(url_data, "is_https", False)))
    features["is_shortened_url"] = int(bool(_safe_get(shortened_url, "detected", False)))
    features["redirect_count"] = int(_safe_get(redirect, "redirect_count", 0) or 0)
    features["has_redirect_loop"] = int(bool(_safe_get(redirect, "loop_detected", False)))
    features["rule_flag_count"] = len(_safe_get(rule_based, "flags", []) or [])
    features["suspicious_param_count"] = len(_safe_get(rule_based, "suspicious_query_params", []) or [])
    features["embedded_url_count"] = len(_safe_get(rule_based, "embedded_urls", []) or [])

    # G6 : external/verification feature
    features["certificate_valid"] = int(bool(_safe_get(certificate, "valid", False)))
    features["safe_browsing_threat"] = int(bool(_safe_get(safe_browsing, "is_threat", False)))

    return {column: features.get(column, 0) for column in FEATURE_COLUMNS}