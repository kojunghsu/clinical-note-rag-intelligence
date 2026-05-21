import os
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import sys
sys.stderr = open(os.devnull, 'w')

import warnings
warnings.filterwarnings("ignore")
import html
import re

import streamlit as st
from src.pipeline import run_pipeline
from src.ingest import build_index
from src.cache import clear_response_cache, init_cache
from src.config import config
from src.retriever import retrieve
from src.reranker import rerank
import time

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ClinicalRAG — MIMIC-IV",
    layout="wide",
    page_icon="⚕",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

/* ── Global Reset ── */
*, *::before, *::after { box-sizing: border-box; }

/*
 * Notion AI-inspired monochrome palette
 * --bg-main:    #FAFAFA  near-white clinical background
 * --bg-sidebar: #F4F4F2  slightly deeper sidebar
 * --bg-card:    #FFFFFF  pure white for elevated cards
 * --bg-input:   #F1F1EF  slightly deeper input zone
 * --border:     #E8E8E5  hairline borders
 * --text-1:     #1A1A1A  near-black primary text
 * --text-2:     #52525B  secondary labels
 * --text-3:     #A1A1AA  muted/meta
 * --text-4:     #D4D4D8  ghost text
 */

.stApp {
    background: #FAFAFA;
    font-family: 'IBM Plex Sans', sans-serif;
    color: #1A1A1A;
}

/* Hide Streamlit dev status banner / toolbar chrome */
[data-testid="stStatusWidget"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stHeader"] {
    display: none !important;
    height: 0 !important;
}

header {
    display: none !important;
    height: 0 !important;
}

/* Kill Streamlit's default top padding */
[data-testid="stAppViewContainer"] > .main > .block-container {
    padding-top: 0px !important;
    padding-left: 0px !important;
    padding-right: 0px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #F4F4F2;
    border-right: 1px solid #E8E8E5;
    padding: 0;
    min-width: 300px !important;
    max-width: 300px !important;
}

/* Hide Streamlit's built-in sidebar collapse button since the sidebar is fixed open. */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"],
button[title="Close sidebar"],
button[title="Open sidebar"] {
    display: none !important;
    visibility: hidden !important;
    pointer-events: none !important;
}

/* Force sidebar visible even if Streamlit remembered a collapsed state */
[data-testid="stSidebar"][aria-expanded="false"] {
    min-width: 300px !important;
    max-width: 300px !important;
    transform: translateX(0) !important;
    margin-left: 0 !important;
    visibility: visible !important;
}

[data-testid="stSidebar"] .block-container {
    padding: 0 !important;
}

.sidebar-header {
    padding: 24px 20px 10px;
}

.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 4px;
}

.sidebar-logo-icon {
    width: 32px;
    height: 32px;
    background: #3F3F46;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 16px;
    flex-shrink: 0;
}

.sidebar-title {
    font-size: 15px;
    font-weight: 600;
    color: #1A1A1A;
    letter-spacing: -0.2px;
}

.sidebar-subtitle {
    font-size: 11px;
    color: #A1A1AA;
    margin-top: 2px;
    letter-spacing: 0.3px;
}

.sidebar-section {
    padding: 16px 20px 8px;
}

.sidebar-section-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #A1A1AA;
    margin-bottom: 10px;
}

/* History items */
.history-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 10px 12px;
    border-radius: 6px;
    cursor: pointer;
    margin-bottom: 2px;
    border: 1px solid transparent;
    transition: all 0.12s ease;
}

.history-item:hover {
    background: #EBEBEA;
    border-color: #E0E0DE;
}

.history-item.active {
    background: #EBEBEA;
    border-color: #D4D4D2;
}

.history-dot {
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: #52525B;
    flex-shrink: 0;
    opacity: 0.5;
    margin-top: 5px;
}

.history-text {
    font-size: 12px;
    color: #52525B;
    line-height: 1.5;
    overflow: hidden;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
}

.history-time {
    font-size: 10px;
    color: #A1A1AA;
    margin-top: 2px;
}

.history-empty {
    font-size: 12px;
    color: #A1A1AA;
    line-height: 1.6;
    padding: 4px 0 2px;
}

/* ── Main content area ── */
.main-container {
    max-width: 860px;
    margin: 0 auto;
    padding: 8px 24px 120px;
}

/* ── Top bar ── */
.top-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 24px;
}

.top-bar-title {
    font-size: 26px;
    font-weight: 600;
    color: #1A1A1A;
    letter-spacing: -0.5px;
}

.top-bar-badge {
    font-size: 12px;
    background: #F4F4F2;
    color: #52525B;
    border: 1px solid #E8E8E5;
    border-radius: 20px;
    padding: 4px 12px;
    font-weight: 500;
}

/* ── Welcome state ── */
.welcome-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 80px 20px 32px;
}

.welcome-icon {
    width: 48px;
    height: 48px;
    background: #F4F4F2;
    border: 1px solid #E8E8E5;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    margin-bottom: 20px;
    color: #52525B;
}

.welcome-heading {
    font-size: 26px;
    font-weight: 600;
    color: #1A1A1A;
    letter-spacing: -0.6px;
    margin-bottom: 10px;
}

.welcome-subtext {
    font-size: 15px;
    color: #A1A1AA;
    font-weight: 400;
    letter-spacing: -0.1px;
}

/* ── Chat messages ── */
.chat-scroll {
    margin-bottom: 24px;
}

/* User message bubble — light gray, consistent with sidebar */
.msg-user {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 16px;
}

.msg-user-bubble {
    background: #EBEBEA;
    color: #1A1A1A;
    border: 1px solid #E0E0DE;
    padding: 11px 17px;
    border-radius: 16px 16px 4px 16px;
    font-size: 15px;
    line-height: 1.6;
    max-width: 72%;
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Assistant message */
.msg-assistant {
    display: flex;
    gap: 12px;
    margin-bottom: 20px;
    align-items: flex-start;
}

.msg-avatar {
    width: 28px;
    height: 28px;
    border-radius: 7px;
    background: #F4F4F2;
    border: 1px solid #E8E8E5;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 13px;
    flex-shrink: 0;
    margin-top: 2px;
    color: #52525B;
}

.msg-assistant-content {
    flex: 1;
}

/* Answer card — pure white, elevated */
.answer-card {
    background: #FFFFFF;
    border: 1px solid #E8E8E5;
    border-radius: 8px;
    padding: 16px 18px;
    font-size: 15px;
    line-height: 1.6;
    color: #1A1A1A;
    margin-bottom: 10px;
    font-family: 'IBM Plex Sans', sans-serif;
    word-wrap: break-word;
    overflow-wrap: break-word;
    overflow-x: hidden;
}

.answer-claim {
    padding: 9px 0;
    border-bottom: 1px solid #F4F4F2;
}

.answer-claim:first-child {
    padding-top: 0;
}

.answer-claim:last-child {
    border-bottom: 0;
    padding-bottom: 0;
}

.cite-chip {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 24px;
    height: 22px;
    border-radius: 999px;
    border: 1px solid #E4E4E7;
    background: #F4F4F2;
    color: #3F3F46;
    font-size: 11px;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    margin-left: 6px;
    vertical-align: 1px;
}

/* Meta row */
.meta-row {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
    margin-bottom: 10px;
}

.badge {
    font-size: 11px;
    border-radius: 4px;
    padding: 3px 10px;
    font-weight: 500;
    border: 1px solid;
    white-space: nowrap;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.1px;
}

/* All badges use the same neutral monochrome system */
.badge-blue {
    background: #F4F4F2;
    color: #3F3F46;
    border-color: #E4E4E7;
}

.badge-green {
    background: #F0FDF4;
    color: #166534;
    border-color: #BBF7D0;
}

.badge-orange {
    background: #FFF7ED;
    color: #9A3412;
    border-color: #FED7AA;
}

.badge-gray {
    background: #F4F4F2;
    color: #71717A;
    border-color: #E4E4E7;
}

.ragas-panel {
    border: 1px solid #E4E4E7;
    background: #FAFAFA;
    border-radius: 7px;
    padding: 9px 11px;
    margin: 0 0 12px;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 6px;
}

.ragas-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #52525B;
    font-family: 'IBM Plex Mono', monospace;
    margin-right: 3px;
}

.ragas-metric,
.ragas-overall {
    font-size: 10.5px;
    color: #3F3F46;
    border-radius: 4px;
    padding: 3px 8px;
    font-family: 'IBM Plex Mono', monospace;
    white-space: nowrap;
    border: 1px solid;
}

.ragas-faithfulness {
    background: #EEEAF5;
    border-color: #D9D0E8;
}

.ragas-relevancy {
    background: #E9EEF0;
    border-color: #D4DEE2;
}

.ragas-overall {
    font-weight: 600;
}

.ragas-overall-low {
    background: #F4E7E5;
    border-color: #E5C9C5;
    color: #7A4A45;
}

.ragas-overall-medium {
    background: #E7EEF5;
    border-color: #CAD8E6;
    color: #455D73;
}

.ragas-overall-high {
    background: #E8F2EA;
    border-color: #CDE1D1;
    color: #496B50;
}

/* Rewrite row */
.rewrite-row {
    font-size: 11.5px;
    color: #71717A;
    font-style: italic;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 6px;
}

.rewrite-arrow {
    color: #3F3F46;
    font-style: normal;
    font-size: 11px;
    opacity: 0.7;
}

/* Sources */
.sources-row {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-bottom: 10px;
}

.source-pill {
    background: #F4F4F2;
    border: 1px solid #E4E4E7;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 10.5px;
    color: #71717A;
    font-family: 'IBM Plex Mono', monospace;
}

/* Retrieved chunks expander */
.chunk-card {
    background: #FAFAFA;
    border: 1px solid #E8E8E5;
    border-radius: 7px;
    padding: 11px 14px;
    margin-bottom: 6px;
}

.chunk-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 5px;
}

.chunk-tag {
    font-size: 9.5px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #3F3F46;
    font-family: 'IBM Plex Mono', monospace;
}

.chunk-score {
    font-size: 10.5px;
    color: #A1A1AA;
    font-family: 'IBM Plex Mono', monospace;
}

.chunk-meta {
    font-size: 10.5px;
    color: #A1A1AA;
    margin-bottom: 5px;
    font-family: 'IBM Plex Mono', monospace;
}

.chunk-content {
    font-size: 12px;
    color: #52525B;
    line-height: 1.65;
}

/* ── Input bar ── */
.input-wrapper {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: linear-gradient(to top, #F1F1EF 82%, transparent);
    padding: 16px 24px 24px;
    z-index: 100;
}

.input-inner {
    max-width: 860px;
    margin: 0 auto;
}

/* Target both text_input and text_area since Streamlit version may vary */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background: #FFFFFF !important;
    border: 1px solid #E0E0DE !important;
    border-radius: 10px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 14px !important;
    color: #1A1A1A !important;
    padding: 11px 15px !important;
    resize: none !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05) !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
    caret-color: #0B1F3A !important;
}

.chat-input-container [data-testid="stTextArea"],
.chat-input-container [data-testid="stTextArea"] > div,
.chat-input-container [data-testid="stTextArea"] > div > div,
.st-key-chat_input_container [data-testid="stTextArea"],
.st-key-chat_input_container [data-testid="stTextArea"] > div,
.st-key-chat_input_container [data-testid="stTextArea"] > div > div {
    min-height: 46px !important;
    height: 46px !important;
}

.chat-input-container [data-testid="stTextArea"] textarea,
.st-key-chat_input_container [data-testid="stTextArea"] textarea {
    min-height: 46px !important;
    height: 46px !important;
    line-height: 22px !important;
    overflow-y: hidden !important;
}

[data-baseweb="textarea"],
[data-baseweb="textarea"] > div {
    border-color: #E0E0DE !important;
    box-shadow: none !important;
    outline: none !important;
}

[data-testid="stTextArea"] textarea:invalid,
[data-testid="stTextArea"] textarea[aria-invalid="true"],
[data-baseweb="textarea"][aria-invalid="true"],
[data-baseweb="textarea"] [aria-invalid="true"] {
    border-color: #0B1F3A !important;
    box-shadow: 0 0 0 2px rgba(11,31,58,0.14) !important;
    outline: none !important;
}

[data-testid="stTextInput"] input:focus,
[data-testid="stTextInput"] input:focus-visible,
[data-testid="stTextArea"] textarea:focus,
[data-testid="stTextArea"] textarea:focus-visible,
[data-baseweb="textarea"]:focus-within,
[data-baseweb="textarea"] > div:focus-within {
    border-color: #0B1F3A !important;
    box-shadow: 0 0 0 2px rgba(11,31,58,0.14) !important;
    outline: none !important;
    outline-color: #0B1F3A !important;
}

/* Streamlit button tweaks */
[data-testid="stButton"] > button {
    border-radius: 8px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    transition: all 0.12s !important;
}

[data-testid="stButton"] > button[kind="primary"] {
    background: #EBEBEA !important;
    border-color: #D4D4D2 !important;
    color: #3F3F46 !important;
}

[data-testid="stButton"] > button[kind="primary"]:hover {
    background: #E0E0DE !important;
    border-color: #C8C8C6 !important;
    color: #1A1A1A !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.07) !important;
}

/* Hide streamlit branding */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

[data-testid="stSidebar"][aria-expanded="true"] {
    min-width: 300px !important;
    max-width: 300px !important;
}

[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1px solid #E8E8E5 !important;
    border-radius: 8px !important;
    overflow: hidden !important;
    margin-bottom: 14px !important;
}

[data-testid="stExpander"] summary {
    font-size: 12px !important;
    color: #3F3F46 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
}

/* Sidebar buttons */
[data-testid="stSidebar"] [data-testid="stButton"] > button {
    width: 100%;
    text-align: left !important;
    background: transparent !important;
    border: 1px solid #E8E8E5 !important;
    color: #52525B !important;
    padding: 7px 11px !important;
    border-radius: 6px !important;
    font-size: 12.5px !important;
}

[data-testid="stSidebar"] [data-testid="stButton"] > button:hover {
    background: #EBEBEA !important;
    border-color: #D4D4D2 !important;
    color: #1A1A1A !important;
}

.sidebar-trash {
    position: fixed !important;
    left: 20px !important;
    bottom: 18px !important;
    width: 36px !important;
    min-width: 36px !important;
    height: 36px !important;
    padding: 0 !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
    border-radius: 8px !important;
    border: 1px solid #E8E8E5 !important;
    background: #F4F4F2 !important;
    color: #52525B !important;
    font-size: 15px !important;
    text-decoration: none !important;
    z-index: 250 !important;
    transition: all 0.12s ease !important;
}

.sidebar-trash:hover {
    background: #EBEBEA !important;
    border-color: #D4D4D2 !important;
    color: #1A1A1A !important;
}

.sidebar-trash-tip {
    display: none;
    position: absolute;
    left: 44px;
    bottom: 2px;
    white-space: nowrap;
    background: #FFFFFF;
    border: 1px solid #E0E0DE;
    border-radius: 8px;
    color: #2F3437;
    font-size: 12px;
    line-height: 1;
    padding: 10px 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    pointer-events: none;
}

.sidebar-trash:hover .sidebar-trash-tip {
    display: block;
}

/* Scrollbar */
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #D4D4D8; border-radius: 3px; }

/* Spinner */
[data-testid="stSpinner"] {
    color: #52525B !important;
}

/* Success / warning */
[data-testid="stAlert"] {
    border-radius: 7px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 13px !important;
}

/* Divider */
hr { border-color: #E8E8E5; }

/* ── Evidence Vault — right column panel ── */

/* Wrapper gives the column consistent padding */
.ev-col-wrapper {
    border-left: 0;
    padding: 0 4px 140px 28px;
    margin-top: -68px !important;
}

/* Vault header block */
.ev-panel-hdr {
    padding: 0 0 12px 0;
    border-bottom: 1px solid #E8E8E5;
    margin-bottom: 12px;
    margin-top: 0 !important
}

.ev-panel-lbl {
    font-size: 9px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #A1A1AA;
    font-family: 'IBM Plex Mono', monospace;
    margin-bottom: 3px;
}

.ev-panel-title {
    font-size: 13px;
    font-weight: 600;
    color: #1A1A1A;
    letter-spacing: -0.2px;
}

.ev-count-line {
    margin-top: 10px;
    font-size: 11px;
    color: #71717A;
    font-family: 'IBM Plex Mono', monospace;
}

.ev-query-line {
    margin-top: 5px;
    font-size: 11px;
    color: #A1A1AA;
    line-height: 1.45;
    font-style: italic;
}

/* Evidence cards list */
.ev-cards-wrap {
    display: flex;
    flex-direction: column;
    gap: 14px;
}

/* Individual evidence card */
.ev-card {
    border: 1px solid #E8E8E5;
    border-radius: 8px;
    overflow: hidden;
    background: #FFFFFF;
    transition: border-color 0.15s, box-shadow 0.15s;
}

.ev-card:hover {
    border-color: #D4D4D2;
    box-shadow: 0 1px 5px rgba(0,0,0,0.05);
}

/* Card header area */
.ev-head {
    padding: 8px 11px 7px;
    background: #FAFAFA;
    border-bottom: 1px solid #F0F0EE;
}

.ev-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 2px;
}

.ev-refid {
    font-size: 10px;
    font-family: 'IBM Plex Mono', monospace;
    color: #2F3437;
    font-weight: 600;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 100%;
}

.ev-noteid {
    font-size: 10px;
    font-family: 'IBM Plex Mono', monospace;
    color: #52525B;
    font-weight: 500;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 100%;
}

/* Card body: content snippet */
.ev-body { padding: 10px 11px 12px; }

.ev-snippet {
    font-size: 11.8px;
    color: #52525B;
    line-height: 1.65;
    word-break: break-word;
}

.ev-highlight {
    background: #FFF3B0;
    color: #27272A;
    border-radius: 3px;
    padding: 0 2px;
}

.ev-tone-1 .ev-head { background: #ECE7DF; border-bottom-color: #DDD4C9; }
.ev-tone-2 .ev-head { background: #DFE8E4; border-bottom-color: #CEDBD6; }
.ev-tone-3 .ev-head { background: #E7E4EE; border-bottom-color: #D7D2E2; }
.ev-tone-4 .ev-head { background: #E8E1DD; border-bottom-color: #D9CBC4; }
.ev-tone-5 .ev-head { background: #E5E7DC; border-bottom-color: #D5D8C9; }

/* Empty / waiting state */
.ev-empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 48px 16px;
    border: 1px solid #E8E8E5;
    border-radius: 8px;
    background: #FFFFFF;
}

.ev-empty-icon {
    font-size: 26px;
    opacity: 0.12;
    margin-bottom: 12px;
}

.ev-empty-txt {
    font-size: 12px;
    color: #A1A1AA;
    line-height: 1.75;
    max-width: 185px;
}


section[data-testid="stHorizontalBlock"] div[data-testid="column"] > div {
    margin-top: 0 !important;
    padding-top: 0 !important;
}


.chat-scroll-area {
    flex: 1;
    overflow-y: auto;
}

/* ── Button + input vertical alignment ── */
[data-testid="stButton"] > button {
    height: 46px !important;
    margin-top: 0 !important;
    margin-bottom: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

[data-testid="stTextInput"] {
    height: 42px !important;
}

[data-testid="stTextInput"] > div {
    height: 100% !important;
}

[data-testid="stTextInput"] input {
    height: 100% !important;
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    box-shadow: none !important;
}

/* Align the input+button columns to vertical center */
[data-testid="stHorizontalBlock"] {
    align-items: flex-start !important;
}
[data-testid="column"] {
    display: flex !important;
    flex-direction: column !important;
    justify-content: flex-start !important;
}
[data-testid="column"] > div:first-child {
    margin-top: 0 !important;
    padding-top: 0 !important;
}
.chat-input-container div[data-testid="stHorizontalBlock"],
.st-key-chat_input_container div[data-testid="stHorizontalBlock"] {
    align-items: center !important;
    gap: 12px !important;
}

.chat-input-container [data-testid="column"],
.st-key-chat_input_container [data-testid="column"] {
    display: flex !important;
    justify-content: center !important;
}

.chat-input-container [data-testid="column"] > div,
.st-key-chat_input_container [data-testid="column"] > div {
    margin-top: 0 !important;
    margin-bottom: 0 !important;
    padding-top: 0 !important;
    padding-bottom: 0 !important;
}


</style>
""", unsafe_allow_html=True)

# ─── Init session state ──────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "query_input" not in st.session_state:
    st.session_state.query_input = ""
if "input_key" not in st.session_state:
    st.session_state.input_key = 0

# Evidence Vault state — holds docs from the most recent pipeline run
if "ev_docs" not in st.session_state:
    st.session_state.ev_docs = []
if "ev_query" not in st.session_state:
    st.session_state.ev_query = ""


def display_answer_only(answer: str) -> str:
    """Hide model self-assessed confidence in the UI; evidence ranking is shown separately."""
    if not answer:
        return ""

    marker = "\nConfidence:"
    if marker in answer:
        return answer.split(marker, 1)[0].strip()

    if answer.startswith("Confidence:"):
        return ""

    answer = answer.strip()
    if answer.startswith("Answer:"):
        answer = answer[len("Answer:"):].strip()

    return answer


def format_ragas_scores_html(scores: dict | None) -> str:
    if not scores:
        return ""

    faithfulness = scores.get("faithfulness")
    answer_relevancy = scores.get("answer_relevancy")
    overall = scores.get("overall")

    if faithfulness is None or answer_relevancy is None or not overall:
        return ""

    overall_key = str(overall).strip().lower()
    if overall_key not in {"low", "medium", "high"}:
        overall_key = "medium"

    return f"""
    <div class="ragas-panel">
        <span class="ragas-label">RAGAS</span>
        <span class="ragas-metric ragas-faithfulness">faithfulness {float(faithfulness):.4f}</span>
        <span class="ragas-metric ragas-relevancy">answer_relevancy {float(answer_relevancy):.4f}</span>
        <span class="ragas-overall ragas-overall-{overall_key}">Overall {html.escape(str(overall))}</span>
    </div>
    """


STOPWORDS = {
    "the", "and", "with", "that", "this", "what", "were", "was", "are", "for",
    "from", "into", "about", "between", "patient", "patients", "noted", "does",
    "have", "has", "her", "his", "their", "how", "why", "who", "when", "where",
    "which", "relation", "relationship", "medical", "history", "hospital",
    "course", "current", "condition", "conditions",
}


def extract_terms(text: str, limit: int = 18) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", text.lower())
    terms = []
    for word in words:
        if word not in STOPWORDS and word not in terms:
            terms.append(word)
    return terms[:limit]


def query_terms(query_text: str) -> list[str]:
    return extract_terms(query_text, limit=12)


def sentence_refs(sentence: str, docs: list[dict], max_refs: int = 2) -> list[int]:
    terms = extract_terms(sentence, limit=12)
    if not terms:
        return []

    ranked = []
    for idx, doc in enumerate(docs, start=1):
        content = str(doc.get("content", "")).lower()
        score = sum(1 for term in terms if term in content)
        if score:
            ranked.append((score, idx))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [idx for _, idx in ranked[:max_refs]]


def split_answer_claims(text: str) -> list[str]:
    pieces = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", paragraph)
        pieces.extend(sentence.strip() for sentence in sentences if sentence.strip())
    return pieces or [text.strip()]


def format_answer_html(answer: str, docs: list[dict] | None = None) -> str:
    cleaned = display_answer_only(answer)
    docs = docs or []
    if not cleaned:
        return "<div class='answer-card'><div class='answer-claim'>Unable to confirm from the available retrieved notes.</div></div>"

    claims = []
    for claim in split_answer_claims(cleaned):
        refs = sentence_refs(claim, docs)
        chips = "".join(f"<span class='cite-chip'>[{ref}]</span>" for ref in refs)
        claims.append(f"<div class='answer-claim'>{html.escape(claim)}{chips}</div>")

    return f"<div class='answer-card'>{''.join(claims)}</div>"


def relevant_excerpt(content: str, query_text: str, window: int = 560) -> str:
    text = re.sub(r"\s+", " ", content).strip()
    if len(text) <= window:
        return text

    lowered = text.lower()
    terms = query_terms(query_text)
    positions = [lowered.find(term) for term in terms if lowered.find(term) >= 0]

    if positions:
        center = min(positions)
    else:
        center = 0

    start = max(center - window // 4, 0)
    end = min(start + window, len(text))
    start = max(end - window, 0)

    excerpt = text[start:end].strip()
    if start > 0:
        excerpt = "..." + excerpt
    if end < len(text):
        excerpt = excerpt + "..."
    return excerpt


def highlight_excerpt(excerpt: str, query_text: str) -> str:
    escaped = html.escape(excerpt)
    terms = sorted(query_terms(query_text), key=len, reverse=True)
    for term in terms:
        pattern = re.compile(rf"(?i)(?<![A-Za-z0-9])({re.escape(term)})(?![A-Za-z0-9])")
        escaped = pattern.sub(r"<mark class='ev-highlight'>\1</mark>", escaped)
    return escaped


def hydrate_cached_evidence(query_text: str, result):
    """Cached answers do not store docs; retrieve evidence for display only."""
    if not getattr(result, "cache_hit", False):
        return result

    if result.reranked_docs or result.retrieved_docs:
        return result

    docs = retrieve(query_text, k=config.TOP_K)
    result.retrieved_docs = docs
    result.reranked_docs = rerank(query_text, docs, config.TOP_N) if docs else []
    return result

init_cache()

if st.query_params.get("clear_history_cache") == "1":
    st.session_state.chat_history = []
    st.session_state.ev_docs = []
    st.session_state.ev_query = ""
    st.session_state.input_key += 1
    clear_response_cache()
    st.query_params.clear()
    st.rerun()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo / header
    st.markdown("""
    <div class="sidebar-header">
        <div class="sidebar-logo">
            <div class="sidebar-logo-icon">⚕</div>
            <div>
                <div class="sidebar-title">ClinicalRAG</div>
                <div class="sidebar-subtitle">MIMIC-IV · Discharge Notes</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Index management
    st.markdown('<div class="sidebar-section"><div class="sidebar-section-label">Index</div>', unsafe_allow_html=True)
    if st.button("⟳  Build / Refresh FAISS Index", use_container_width=True):
        os.makedirs(config.FAISS_INDEX_DIR, exist_ok=True)
        with st.spinner("Building index..."):
            build_index()
        st.success("Index built.")

    st.markdown('</div>', unsafe_allow_html=True)

    history_items = [msg for msg in st.session_state.chat_history if msg.get("role") == "user"]
    st.markdown('<div class="sidebar-section"><div class="sidebar-section-label">History</div>', unsafe_allow_html=True)
    if history_items:
        for idx, msg in enumerate(reversed(history_items), start=1):
            text = html.escape(msg.get("content", "Untitled query"))
            timestamp = html.escape(msg.get("timestamp", ""))
            active_class = " active" if idx == 1 else ""
            st.markdown(f"""
            <div class="history-item{active_class}">
                <div class="history-dot"></div>
                <div>
                    <div class="history-text">{text}</div>
                    <div class="history-time">{timestamp}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="history-empty">No queries yet.</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <a class="sidebar-trash" href="?clear_history_cache=1" aria-label="Clear history and cache">
        🗑
        <span class="sidebar-trash-tip">Clear history and cache</span>
    </a>
    """, unsafe_allow_html=True)

# ─── Main content — two-column layout ────────────────────────────────────────
# Left  [5]: chat thread + pinned input bar
# Right [3]: Evidence Vault — shows reranked docs from the latest query
col_chat, col_vault = st.columns([5, 3], gap="small")

# ── LEFT: Chat column ─────────────────────────────────────────────────────────
with col_chat:
    st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
    st.markdown('<div class="chat-scroll-area">', unsafe_allow_html=True)

    st.markdown('<div style="padding: 0 8px 0 24px;">', unsafe_allow_html=True)

    # Top bar
    n_queries = sum(1 for m in st.session_state.chat_history if m["role"] == "user")
    st.markdown(f"""
    <div class="top-bar">
        <div class="top-bar-title">Clinical Insights</div>
        <span class="top-bar-badge">{n_queries} {'query' if n_queries == 1 else 'queries'} this session</span>
    </div>
    """, unsafe_allow_html=True)

    # Welcome
    if not st.session_state.chat_history:
        st.markdown("""
        <div class="welcome-container">
            <div class="welcome-icon">⚕</div>
            <div class="welcome-subtext">Surface clinical patterns from patient notes</div>
        </div>
        """, unsafe_allow_html=True)

    # Chat messages
    else:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="msg-user">
                    <div class="msg-user-bubble">{msg["content"]}</div>
                </div>
                """, unsafe_allow_html=True)

            elif msg["role"] == "assistant":
                result = msg["result"]

                docs_for_count = result.reranked_docs or result.retrieved_docs or []
                n_docs = len(docs_for_count)

                badges = f"<span class='badge badge-blue'>📄 {n_docs} notes</span>"
                if result.cache_hit:
                    badges += "<span class='badge badge-gray'>⚡ cached</span>"
                if result.used_fallback:
                    badges += "<span class='badge badge-orange'>⚠ fallback</span>"

                st.markdown(f"""
                <div class="msg-assistant">
                    <div class="msg-avatar">⚕</div>
                    <div class="msg-assistant-content">
                        <div class="meta-row">{badges}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown('<div style="margin-left: 40px;">', unsafe_allow_html=True)
                st.markdown(format_ragas_scores_html(getattr(result, "ragas_scores", None)), unsafe_allow_html=True)
                st.markdown(format_answer_html(result.answer, docs_for_count), unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)  

    with st.container(key="chat_input_container"):
        input_col, btn_col = st.columns([10, 0.72], gap="small")

        with input_col:
            query = st.text_input(
                "Query",
                placeholder="Ask about patient treatments, diagnoses, rare patterns...",
                label_visibility="collapsed",
                key=f"query_input_{st.session_state.input_key}",
            )

        with btn_col:
            run_clicked = st.button("→")

    st.markdown('</div>', unsafe_allow_html=True) 

# ── RIGHT: Evidence Vault column ───────────────────────────────────────────────
with col_vault:
    st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
    st.markdown('<div class="ev-col-wrapper">', unsafe_allow_html=True)
    st.markdown("""
    """, unsafe_allow_html=True)
    ev_docs  = st.session_state.get("ev_docs", [])
    ev_query = st.session_state.get("ev_query", "")

    # Vault header
    n_ev = len(ev_docs)
    query_line = html.escape(ev_query) if ev_query else "No query yet"
    count_line = "No notes retrieved yet" if not ev_docs else f"{n_ev} retrieved note{'s' if n_ev != 1 else ''}"
    st.markdown(f"""
    <div class="ev-panel-hdr">
        <div class="ev-panel-lbl">Evidence Vault</div>
        <div class="ev-panel-title">Retrieved Notes</div>
        <div class="ev-count-line">{count_line}</div>
        <div class="ev-query-line">{query_line}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="ev-cards-wrap">', unsafe_allow_html=True)

    if not ev_docs:
        # Waiting-state placeholder
        st.markdown("""
            <div class="ev-empty">
                <div class="ev-empty-icon">📋</div>
            <div class="ev-empty-txt">Submit a clinical query to see retrieved notes here.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for i, doc in enumerate(ev_docs, start=1):
            meta         = doc.get("metadata", {})
            note_id_raw  = meta.get("note_id") or meta.get("source") or f"doc_{i}"
            content_raw  = str(doc.get("content", ""))
            excerpt_raw  = relevant_excerpt(content_raw, ev_query)
            tone_cls     = f"ev-tone-{((i - 1) % 5) + 1}"

            # Render user/data text as plain text inside HTML cards to prevent
            # markdown/html injection artifacts in the Evidence Vault.
            ref_id   = html.escape(f"Reference [{i}]")
            note_id  = html.escape(str(note_id_raw))
            snippet  = highlight_excerpt(excerpt_raw, ev_query)

            with st.expander(f"[{i}] {note_id_raw}", expanded=True):
                st.markdown(f"""
                <div class="ev-card {tone_cls}">
                    <div class="ev-head">
                        <div class="ev-top">
                            <span class="ev-refid">{ref_id}</span>
                        </div>
                        <div class="ev-top">
                            <span class="ev-noteid">{note_id}</span>
                        </div>
                    </div>
                    <div class="ev-body">
                        <div class="ev-snippet">{snippet}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # ev-cards-wrap
    st.markdown('</div>', unsafe_allow_html=True)  # ev-col-wrapper


# ── Run pipeline ───────────────────────────────────────────────────────────────
should_run = run_clicked

if should_run and query and query.strip():
    st.session_state["last_submitted"] = query.strip()
    ts = time.strftime("%H:%M")

    # Add user message to history
    st.session_state.chat_history.append({
        "role": "user",
        "content": query.strip(),
        "timestamp": ts
    })

    with st.spinner("Retrieving clinical context…"):
        result = run_pipeline(query.strip())
        result = hydrate_cached_evidence(query.strip(), result)

    # Add assistant message to history
    st.session_state.chat_history.append({
        "role": "assistant",
        "result": result,
        "query": query.strip()
    })

    # Push latest retrieved docs to the Evidence Vault
    st.session_state["ev_docs"]  = result.reranked_docs or result.retrieved_docs or []
    st.session_state["ev_query"] = query.strip()

    # Clear input box by changing its key
    st.session_state.input_key += 1

    st.rerun()

elif run_clicked and (not query or not query.strip()):
    st.warning("Please enter a query before running.")
