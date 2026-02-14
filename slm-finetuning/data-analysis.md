# Dataset Analysis: Banking Voice Assistant SLM (v2)

## Overview

| Metric | Train | Test |
|--------|-------|------|
| Total Conversations | 50 | 50 |
| Format | Complete multi-turn conversations | Complete multi-turn conversations |
| Ending pattern | All end with `goodbye` | All end with `goodbye` |

All single-turn conversations have been removed. Every conversation is 2-5 turns and represents a complete user interaction from start to finish, ending with `thank_you → goodbye` or just `goodbye`.

---

## Conversation Length Distribution

Number of user turns per conversation:

| Turns | Train | Test | Train % | Test % |
|-------|-------|------|---------|--------|
| 2 | 1 | 1 | 2% | 2% |
| 3 | 14 | 13 | 28% | 26% |
| 4 | 26 | 24 | 52% | 48% |
| 5 | 9 | 12 | 18% | 24% |

The majority of conversations are 3-4 turns, with a good representation of 5-turn conversations for complex multi-step interactions.

---

## Function Distribution (All Calls)

Counts include all tool calls across entire conversations (intermediate + final).

| Function | Train | Test | Train % | Test % |
|----------|-------|------|---------|--------|
| goodbye | 50 | 50 | 25.9% | 25.4% |
| thank_you | 32 | 33 | 16.6% | 16.8% |
| greeting | 26 | 28 | 13.5% | 14.2% |
| transfer_money | 14 | 12 | 7.3% | 6.1% |
| check_balance | 13 | 14 | 6.7% | 7.1% |
| cancel_card | 12 | 10 | 6.2% | 5.1% |
| pay_bill | 8 | 10 | 4.1% | 5.1% |
| report_fraud | 9 | 6 | 4.7% | 3.0% |
| get_statement | 6 | 9 | 3.1% | 4.6% |
| reset_pin | 6 | 5 | 3.1% | 2.5% |
| replace_card | 5 | 6 | 2.6% | 3.0% |
| speak_to_human | 5 | 6 | 2.6% | 3.0% |
| activate_card | 4 | 5 | 2.1% | 2.5% |
| intent_unclear | 3 | 3 | 1.6% | 1.5% |

**Coverage:** All 14 functions are represented in both datasets.

---

## Parameter Usage by Function

### Account Functions

**check_balance**
| Parameter | Train | Test |
|-----------|-------|------|
| account_type | 12/13 (92%) | 12/14 (86%) |

**get_statement**
| Parameter | Train | Test |
|-----------|-------|------|
| account_type | 5/6 (83%) | 7/9 (78%) |
| period | 5/6 (83%) | 6/9 (67%) |

### Transfer Functions

**transfer_money**
| Parameter | Train | Test |
|-----------|-------|------|
| amount | 12/14 (86%) | 10/12 (83%) |
| from_account | 10/14 (71%) | 9/12 (75%) |
| to_account | 10/14 (71%) | 11/12 (92%) |

### Card Functions

**cancel_card**
| Parameter | Train | Test |
|-----------|-------|------|
| card_type | 10/12 (83%) | 8/10 (80%) |
| card_last_four | 7/12 (58%) | 6/10 (60%) |
| reason | 7/12 (58%) | 6/10 (60%) |

**replace_card**
| Parameter | Train | Test |
|-----------|-------|------|
| card_type | 4/5 (80%) | 5/6 (83%) |
| card_last_four | 3/5 (60%) | 4/6 (67%) |

**activate_card**
| Parameter | Train | Test |
|-----------|-------|------|
| card_last_four | 3/4 (75%) | 3/5 (60%) |

**report_fraud**
| Parameter | Train | Test |
|-----------|-------|------|
| card_type | 7/9 (78%) | 5/6 (83%) |
| card_last_four | 2/9 (22%) | 2/6 (33%) |
| transaction_amount | 4/9 (44%) | 3/6 (50%) |

**reset_pin**
| Parameter | Train | Test |
|-----------|-------|------|
| card_type | 4/6 (67%) | 3/5 (60%) |
| card_last_four | 2/6 (33%) | 2/5 (40%) |

### Payment Functions

**pay_bill**
| Parameter | Train | Test |
|-----------|-------|------|
| payee | 6/8 (75%) | 8/10 (80%) |
| amount | 6/8 (75%) | 7/10 (70%) |
| from_account | 4/8 (50%) | 3/10 (30%) |

### Support Functions

**speak_to_human**
| Parameter | Train | Test |
|-----------|-------|------|
| department | 4/5 (80%) | 5/6 (83%) |

### Dialogue Control Functions

| Function | Parameters | Train | Test |
|----------|------------|-------|------|
| greeting | (none) | 26 | 28 |
| goodbye | (none) | 50 | 50 |
| thank_you | (none) | 32 | 33 |
| intent_unclear | (none) | 3 | 3 |

---

## Multi-Turn Slot-Filling Patterns

The dataset includes conversations where users provide information incrementally, with the model issuing updated tool calls as more details arrive.

### 2-step slot filling
- `transfer_money({}) → transfer_money({amount, from, to})`
- `cancel_card({type}) → cancel_card({type, last_four, reason})`
- `activate_card({}) → activate_card({last_four})`
- `reset_pin({}) → reset_pin({type})`
- `replace_card({}) → replace_card({type, last_four})`
- `report_fraud({type}) → report_fraud({type, last_four, amount})`
- `get_statement({}) → get_statement({account, period})`
- `pay_bill({}) → pay_bill({payee, amount})`
- `check_balance({}) → check_balance({account_type})`

### 3-step progressive slot filling
- `transfer_money({}) → transfer_money({amount, from}) → transfer_money({amount, from, to})`
- `report_fraud({}) → report_fraud({type}) → report_fraud({type, amount})`
- `cancel_card({}) → cancel_card({type, last_four}) → cancel_card({type, last_four, reason})`
- `pay_bill({}) → pay_bill({payee}) → pay_bill({payee, amount, from})`

### Multi-operation conversations
- `check_balance → transfer_money` (check then move funds)
- `cancel_card → replace_card` (cancel then request replacement)
- `report_fraud → speak_to_human` (report then escalate)
- `pay_bill → get_statement` (pay then request statement)
- `check_balance → check_balance` (check multiple accounts)

### Topic switches
- `cancel_card → check_balance` (user changes mind)
- `report_fraud → check_balance` (user pivots)
- `intent_unclear → greeting → check_balance` (user recovers)

---

## ASR Artifact Coverage

~30% of conversations include realistic speech-to-text transcription artifacts:

| Artifact Type | Examples |
|--------------|----------|
| Filler words | "uh", "um", "like" |
| Word splits | "cred it", "dead it", "save ins", "checkin" |
| Homophones/errors | "ballets" (balance), "stolin" (stolen), "trans fur" (transfer) |
| False starts | "can- cancel" |
| Run-together speech | "endin in four five three two" |

---

## Summary

### Dataset Statistics
- **Total conversations:** 100 (50 train + 50 test)
- **All multi-turn:** 100% (no single-turn examples)
- **Complete conversations:** all end with `goodbye`
- **Function coverage:** 100% (all 14 functions in both splits)
