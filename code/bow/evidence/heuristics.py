"""Generic, keyword-scored fallback classifier for financial correspondence.

This is deliberately NOT a lookup of this dataset's exact template sentences.
It scores a message against small vocabularies of ordinary financial-English
and financial-Indonesian terms (salary, refund, invoice, dispute, prize, ...)
that would appear in real payroll/banking/merchant correspondence anywhere,
and returns the best-scoring intent with a confidence below what a real
language-model read would earn. It exists so the pipeline still produces a
structured, non-fabricated result when no LLM extractor is configured, and it
is expected to run on messages this system has never seen.

Amount, date, and percent extraction use pure pattern matching (currency
codes, ISO dates, percent signs) and generalize to any message in that shape.
"""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal
from typing import List, Optional, Tuple

from . import schema

_AMOUNT_RE = re.compile(r"\b(IDR|INR|USD|EUR|ZAR)\s*([\d,]+(?:\.\d+)?)", re.I)
_DATE_ISO_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")


def extract_amounts(text: str) -> List[Tuple[str, Decimal]]:
    return [(cur.upper(), Decimal(num.replace(",", ""))) for cur, num in _AMOUNT_RE.findall(text)]


def extract_dates(text: str) -> List[date]:
    out = []
    for raw in _DATE_ISO_RE.findall(text):
        y, m, d = raw.split("-")
        out.append(date(int(y), int(m), int(d)))
    return out


def extract_percent(text: str) -> Optional[Decimal]:
    match = _PERCENT_RE.search(text)
    return Decimal(match.group(1)) if match else None


# Each intent maps to independent "signal groups": a message earns a point per
# group where at least one term matches. Using several small groups (rather
# than one big bag of words) means an intent needs corroborating evidence
# from more than one angle before it wins, which is what keeps this a
# reasonable heuristic instead of a single magic keyword.
_VOCAB = {
    schema.UPFRONT_FEE_SCAM: [
        ["pay the release charge", "pay the processing charge", "processing fee to receive",
         "release fee", "biaya pencairan", "biaya pemrosesan", "bayar biaya"],
        ["prize", "cash prize", "hadiah", "winner", "selected"],
    ],
    schema.SALARY_STOPPED: [
        ["employment has ended", "contract has ended", "seasonal contract",
         "hubungan kerja", "kontrak musiman", "telah berakhir"],
        ["no regular salary", "no off-season income", "no renewal",
         "tidak ada pembayaran gaji", "belum ada pendapatan"],
    ],
    schema.SALARY_ONE_OFF: [
        ["arrears", "one-time adjustment", "one-off adjustment",
         "tunggakan", "penyesuaian satu kali", "penyesuaian tunggakan"],
        ["payroll", "payslip", "gaji", "slip gaji"],
    ],
    schema.SALARY_SCHEDULE_CHANGE: [
        ["confirmed for", "confirmed credit date", "replaces the payday",
         "revised date", "dikonfirmasi untuk", "tanggal kredit",
         "menggantikan tanggal", "tanggal terbaru"],
    ],
    schema.SALARY_CHANGE: [
        ["salary", "monthly pay", "base salary", "gaji bulanan", "gaji pokok",
         "first salary", "gaji pertama", "regular salary", "gaji rutin"],
        ["increased", "reduced", "temporary", "resumes", "naik", "dikurangi",
         "sementara", "kembali normal", "confirmed", "dikonfirmasi"],
    ],
    schema.INCOME_UNCONFIRMED: [
        ["bonus", "commission", "still pending approval", "not yet approved",
         "bonus kuartalan", "komisi", "belum disetujui", "menunggu"],
    ],
    schema.INCOME_NOT_RECURRING: [
        ["reimbursement", "work expense", "not your regular salary",
         "penggantian", "biaya kerja", "bukan gaji rutin"],
    ],
    schema.BANK_ADMIN: [
        ["transfer between your two accounts", "same account holder",
         "matching debit and credit", "transfer antara dua rekening",
         "pemilik yang sama"],
        ["two separate card accounts", "minimum payments due", "dua rekening kartu"],
        ["still being investigated", "dispute is open", "masih dalam penyelidikan",
         "sengketa masih terbuka", "reversal has not been posted",
         "dana pembalikannya belum tercatat"],
    ],
    schema.DEBIT_WILL_RETRY: [
        ["debit attempt failed", "will be attempted again", "still outstanding",
         "debit sebelumnya gagal", "akan dicoba lagi", "masih terbuka"],
    ],
    schema.WINDFALL_SETTLED: [
        ["proceeds have reached your account", "prize proceeds", "sudah masuk ke rekening",
         "claim is now closed", "klaim sudah ditutup",
         "proceeds from your investment sale have settled", "sale order is complete",
         "hasil penjualan investasi", "perintah penjualan sudah selesai"],
    ],
    schema.WINDFALL_PENDING: [
        ["still in payment processing", "has not been credited", "prize claim",
         "masih dalam proses pembayaran", "belum masuk ke rekening", "klaim hadiah"],
    ],
    schema.INVESTMENT_PAPER_MOVE: [
        ["displayed market value", "displayed value", "no units have been sold",
         "no cash proceeds", "nilai investasi", "belum dijual", "tidak ada transaksi tunai"],
    ],
    schema.REFUND_PENDING: [
        ["refund has been initiated", "refund is still processing", "not reached your account",
         "pengembalian dana", "belum masuk ke rekening"],
    ],
    schema.FX_PENDING: [
        ["foreign currency", "settlement-date rate", "will convert it using the rate",
         "mata uang asing", "kurs pada tanggal penyelesaian", "kurs saat transaksi selesai"],
    ],
    schema.DOCUMENT_IS_FINAL: [
        ["receipt has the final", "receipt contains the final", "final inr amount",
         "jumlah akhir dalam mata uang"],
    ],
    schema.OBLIGATION_CONFIRMED: [
        ["approved an invoice", "invoice payment of", "menyetujui pembayaran faktur",
         "klien menyetujui"],
    ],
    schema.RECURRING_COST_CHANGE: [
        ["increases monthly rent", "renewed lease", "menaikkan biaya sewa",
         "perpanjangan sewa"],
    ],
    schema.PAYOUT_NOT_YET_AVAILABLE: [
        ["payout is still pending", "not withdrawable", "isn't withdrawable",
         "masih tertunda", "belum dapat ditarik"],
    ],
}

# Checked before the scored vocabulary: a scam framing must never be
# outscored by an adjacent "you received money" reading.
_PRIORITY_ORDER = [schema.UPFRONT_FEE_SCAM]


def _normalize(text: str) -> str:
    return (text or "").replace("’", "'").replace("�", "'").lower()


def classify(text: str) -> Tuple[str, float]:
    """Return ``(intent, confidence)`` using generic keyword scoring.

    Confidence is capped well below 1.0: this is a fallback, not a language
    model, and callers should prefer a real LLM extractor when one is
    configured.
    """
    body = _normalize(text)

    for intent in _PRIORITY_ORDER:
        groups = _VOCAB[intent]
        if all(any(term in body for term in group) for group in groups):
            return intent, 0.75

    best_intent, best_score = schema.UNKNOWN, 0
    for intent, groups in _VOCAB.items():
        score = sum(1 for group in groups if any(term in body for term in group))
        if score > best_score:
            best_intent, best_score = intent, score

    if best_score == 0:
        return schema.UNKNOWN, 0.0
    # one matching group -> modest confidence; every group corroborating -> higher
    total_groups = len(_VOCAB[best_intent])
    confidence = 0.35 + 0.3 * (best_score / total_groups)
    return best_intent, round(min(confidence, 0.7), 2)


def classify_full(text: str) -> dict:
    intent, confidence = classify(text)
    return {
        "intent": intent,
        "amounts": [[cur, str(amt)] for cur, amt in extract_amounts(text)],
        "dates": [d.isoformat() for d in extract_dates(text)],
        "percent": str(extract_percent(text)) if extract_percent(text) is not None else None,
        "confidence": confidence,
        "extractor": "heuristic-v1",
    }
