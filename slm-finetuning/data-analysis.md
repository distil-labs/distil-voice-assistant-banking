# Dataset Analysis: Banking Voice Assistant SLM

## Overview

| Metric | Train | Test |
|--------|-------|------|
| Total Examples | 62 | 61 |

---

## Function Distribution

| Function | Train | Test | Train % | Test % |
|----------|-------|------|---------|--------|
| check_balance | 8 | 7 | 12.9% | 11.5% |
| transfer_money | 7 | 7 | 11.3% | 11.5% |
| pay_bill | 6 | 7 | 9.7% | 11.5% |
| cancel_card | 5 | 5 | 8.1% | 8.2% |
| get_statement | 5 | 5 | 8.1% | 8.2% |
| speak_to_human | 5 | 5 | 8.1% | 8.2% |
| replace_card | 4 | 4 | 6.5% | 6.6% |
| report_fraud | 4 | 4 | 6.5% | 6.6% |
| reset_pin | 4 | 4 | 6.5% | 6.6% |
| activate_card | 3 | 3 | 4.8% | 4.9% |
| greeting | 3 | 2 | 4.8% | 3.3% |
| intent_unclear | 3 | 3 | 4.8% | 4.9% |
| goodbye | 3 | 3 | 4.8% | 4.9% |
| thank_you | 2 | 2 | 3.2% | 3.3% |

**Coverage:** All 14 functions are represented in both datasets.

---

## Conversation Length Distribution

Number of user turns per conversation:

| Turns | Train | Test | Train % | Test % |
|-------|-------|------|---------|--------|
| 1 (single-turn) | 40 | 37 | 64.5% | 60.7% |
| 2 (two-turn) | 18 | 20 | 29.0% | 32.8% |
| 3 (three-turn) | 4 | 4 | 6.5% | 6.6% |

**Improvement:** Multi-turn examples increased from ~25% to ~35% of the dataset, with 3-turn conversations now at 6.5% (up from 2%).

---

## User Message Length Statistics

Character count of the final user message in each conversation:

| Statistic | Train | Test |
|-----------|-------|------|
| Min | 3 | 10 |
| Max | 68 | 55 |
| Mean | 29.7 | 27.0 |
| Median | 28.5 | 24.0 |

---

## Parameter Usage by Function

### Account Functions

**check_balance**
| Parameter | Train | Test |
|-----------|-------|------|
| account_type | 6/8 (75%) | 5/7 (71%) |

**get_statement**
| Parameter | Train | Test |
|-----------|-------|------|
| account_type | 5/5 (100%) | 5/5 (100%) |
| period | 4/5 (80%) | 4/5 (80%) |

### Transfer Functions

**transfer_money**
| Parameter | Train | Test |
|-----------|-------|------|
| amount | 7/7 (100%) | 7/7 (100%) |
| from_account | 7/7 (100%) | 7/7 (100%) |
| to_account | 7/7 (100%) | 7/7 (100%) |

### Card Functions

**cancel_card**
| Parameter | Train | Test |
|-----------|-------|------|
| card_type | 5/5 (100%) | 5/5 (100%) |
| card_last_four | 4/5 (80%) | 4/5 (80%) |
| reason | 2/5 (40%) | 2/5 (40%) |

**replace_card**
| Parameter | Train | Test |
|-----------|-------|------|
| card_type | 3/4 (75%) | 4/4 (100%) |
| card_last_four | 2/4 (50%) | 2/4 (50%) |

**activate_card**
| Parameter | Train | Test |
|-----------|-------|------|
| card_last_four | 2/3 (67%) | 2/3 (67%) |

**report_fraud**
| Parameter | Train | Test |
|-----------|-------|------|
| card_type | 4/4 (100%) | 4/4 (100%) |
| card_last_four | 1/4 (25%) | 1/4 (25%) |
| transaction_amount | 3/4 (75%) | 3/4 (75%) |

**reset_pin**
| Parameter | Train | Test |
|-----------|-------|------|
| card_type | 3/4 (75%) | 3/4 (75%) |
| card_last_four | 2/4 (50%) | 2/4 (50%) |

### Payment Functions

**pay_bill**
| Parameter | Train | Test |
|-----------|-------|------|
| payee | 6/6 (100%) | 7/7 (100%) |
| amount | 5/6 (83%) | 6/7 (86%) |
| from_account | 1/6 (17%) | 2/7 (29%) |

### Support Functions

**speak_to_human**
| Parameter | Train | Test |
|-----------|-------|------|
| department | 3/5 (60%) | 3/5 (60%) |

### Dialogue Control Functions

| Function | Parameters | Train | Test |
|----------|------------|-------|------|
| greeting | (none) | 3 | 2 |
| goodbye | (none) | 3 | 3 |
| thank_you | (none) | 2 | 2 |
| intent_unclear | (none) | 3 | 3 |

---

## Multi-Turn Slot-Filling Patterns

The dataset includes various multi-turn patterns:

### Pattern: User provides incomplete info → Model generates partial call → User completes
- transfer_money: amount only → add accounts
- transfer_money: destination only → add amount and source
- cancel_card: card_type only → add last_four
- pay_bill: amount only → add payee
- pay_bill: payee only → add amount
- report_fraud: generic → add card_type → add amount
- reset_pin: generic → add card_type → add last_four
- get_statement: period only → add account_type
- replace_card: last_four only → add card_type

### Pattern: 3-turn progressive slot filling
- transfer_money: generic → amount + source → destination
- report_fraud: generic → card_type → transaction_amount
- reset_pin: generic → card_type → card_last_four
- get_statement: generic → account_type → period

---

## Summary

### Dataset Statistics
- **Total examples:** 123 (62 train + 61 test)
- **Multi-turn ratio:** ~35% (up from ~25%)
- **3-turn examples:** 8 total (6.5%)
- **Function coverage:** 100% (all 14 functions)

### Strengths
- All 14 functions covered in both train and test sets
- Good balance between train and test distributions
- Strong representation of multi-turn slot-filling conversations
- Variety of parameter combinations (full, partial, empty)
- Progressive slot-filling patterns (1→2→3 turns)

### Potential Future Improvements
- Add more examples for `greeting`, `thank_you` (currently lowest counts)
- Add edge cases with unusual phrasings or ASR errors
- Consider adding more `from_account` examples for `pay_bill`
