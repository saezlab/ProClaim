# Virtual Environment Dependency Analysis Plan

## Issue Summary

測試 repo (`grn-llm-correct-prompt-test`) 的 NLP model loading 有問題，出現以下錯誤：

```
Failed to compute NLP features for PMID 40319835: Unable to compare versions for numpy>=1.17: need=1.17 found=None. This is unusual. Consider reinstalling numpy.
```

## Investigation Findings

### 1. Multiple Virtual Environments in Original Repo

原始 repo (`grn-llm-correct`) 有**三個 virtual environments**:
- `.venv` (Python 3.12.12) - 最近更新於 Mar 10
- `.venv-py312` (Python 3.12.12) - 更新於 Mar 9
- `.venv310` (Python 3.10.19) - 更新於 Feb 18

### 2. Package Version Comparison

#### Original Repo (.venv):
- numpy: 2.4.2 (單一版本)
- spacy: 3.8.11
- scispacy: 0.6.2
- **spacy model works correctly** (有 warnings 但可以運行)

#### Test Repo (.venv):
- numpy: **1.26.4 AND 2.4.2** (雙重安裝！)
- spacy: 3.8.11
- spacy-legacy: 3.0.12
- spacy-loggers: 1.0.5
- **No scispacy listed** (可能未正確安裝)

### 3. Key Differences

| Aspect | Original Repo | Test Repo | Issue |
|--------|---------------|-----------|-------|
| numpy versions | 1 (2.4.2) | 2 (1.26.4 + 2.4.2) | ✗ Conflict |
| scispacy | 0.6.2 | Not listed | ✗ Missing |
| venv count | 3 (.venv, .venv-py312, .venv310) | 1 (.venv) | - |
| spacy model | Works | Works standalone | - |
| Feature extraction | Works | **Fails** | ✗ Main Issue |

### 4. Error Root Cause Analysis

錯誤發生在 `feature_tools.py` 的 `compute_entity_coverage()` 函數中，當 spaCy 處理文本時：

```python
def _get_spacy_nlp():
    """Load spaCy NLP model (singleton, loaded on first call)."""
    global _spacy_nlp
    if _spacy_nlp is None:
        import spacy
        try:
            _spacy_nlp = spacy.load("en_core_sci_sm")  # ← Error occurs here
        except OSError:
            ...
```

**可能的原因**：
1. `scispacy` 模型需要正確的 numpy 版本
2. 雙重 numpy 安裝導致版本檢測失敗
3. scispacy 可能依賴特定的 numpy C API，而雙重安裝破壞了這個關聯

### 5. Why Did `uv sync` Fail?

當你從原始 repo 複製 `uv.lock` 並執行 `uv sync` 時失敗，可能原因：
- `uv.lock` 可能是針對特定 venv 生成的
- 原始 repo 可能使用了不同的 Python 版本或 venv
- `uv.lock` 可能有 platform-specific dependencies

## Proposed Solutions

### Option 1: Clean Reinstall (推薦)
1. 完全刪除 `.venv`
2. 重新運行 `uv sync`
3. 手動安裝 scispacy model：
   ```bash
   uv pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
   ```
4. 驗證只有單一 numpy 版本

### Option 2: Use Original Repo's Specific venv
1. 確認原始 repo 使用哪個 venv (`.venv`, `.venv-py312`, 或 `.venv310`)
2. 檢查該 venv 的完整 package list
3. 在測試 repo 複製相同配置

### Option 3: Manual Dependency Fix
1. 在測試 repo 中卸載所有 numpy 版本：
   ```bash
   uv pip uninstall numpy
   ```
2. 重新安裝單一版本：
   ```bash
   uv pip install "numpy>=2.0,<3.0"
   ```
3. 確保 scispacy 正確安裝

### Option 4: Check for Hidden Dependencies
1. 檢查是否有其他配置文件（如 `.env`, `setup.cfg`, `setup.py`）
2. 檢查是否有 conda 環境混入
3. 檢查 `PYTHONPATH` 環境變數

## Testing Strategy

After implementing the fix, test:
1. ✓ spacy model loads without errors
2. ✓ numpy version check passes
3. ✓ `compute_entity_coverage()` works correctly
4. ✓ Run the failing script again:
   ```bash
   uv run python experiments/run_signor_eval.py
   ```
5. ✓ Check that NLP features compute successfully for all PMIDs

## Questions to Discuss

1. **Which venv does the original repo actually use?**
   - Is it `.venv`, `.venv-py312`, or `.venv310`?
   - When running code, which Python interpreter is being used?

2. **Why multiple venvs in original repo?**
   - Are they for different experiments?
   - Do they have different dependencies?
   - Should we replicate the same structure?

3. **What's the correct workflow?**
   - Should we use `uv sync` or manual `uv pip install`?
   - Is there a specific order for installing dependencies?
   - Are there any post-install scripts?

4. **Environment-specific issues?**
   - Is the HPC environment causing issues?
   - Are there module loads (`module load python`) that affect the setup?
   - Does the original repo have any special environment setup scripts?

## Next Steps

Please review this analysis and let me know:
1. Which solution approach you prefer (Option 1-4)
2. Answers to the questions above
3. Whether we should investigate the original repo's venv setup first
4. If there are any other known issues or workarounds you've tried

---

## SOLUTION FOUND ✅

**Date**: 2026-03-27 (updated)
**Status**: RESOLVED

### Root Cause

The issue was caused by **installation order and dependency conflicts**:

1. `uv sync` installs packages with numpy 2.4.2 and spacy 3.8.11
2. Installing `en-core-sci-sm` model **without `--no-deps`** triggers downgrades:
   - spacy 3.8.11 → 3.7.5 (because model was trained with 3.7.4)
   - numpy 2.4.x → 1.26.4
3. Installing `scispacy==0.6.2` downgrades numpy 2.4.x → 1.26.4 (due to `nmslib-metabrainz` dependency)

### Correct Installation Order

The key is to install the scispacy model **with `--no-deps` flag FIRST**, before installing scispacy:

```bash
# 1. Clean install
rm -rf .venv
uv sync  # → installs spacy 3.8.11, numpy 2.4.2

# 2. Install model WITHOUT dependencies (prevents downgrades)
uv pip install --no-deps https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz

# 3. Install scispacy (will downgrade numpy to 1.26.4)
uv pip install scispacy==0.6.2

# 4. Upgrade numpy back to 2.4+
uv pip install "numpy>=2.4,<3.0"
```

### Final Working Configuration

- ✅ Python 3.12.12
- ✅ numpy 2.4.3
- ✅ spacy 3.8.11
- ✅ scispacy 0.6.2
- ✅ en-core-sci-sm 0.5.4
- ✅ blis 1.3.3
- ✅ thinc 8.3.10

### Testing Results

```python
from pkevolve.verification.feature_tools import compute_entity_coverage
result = compute_entity_coverage('EGFR activates MAPK', 'EGFR protein activates MAPK pathway')
# Result: claim_entity_coverage=0.0 claim_entities=['egfr', 'mapk'] evidence_entities=['egfr protein', 'mapk pathway']
```

✅ **NLP features compute successfully**
⚠️ Version warning appears but does not affect functionality:
```
UserWarning: [W095] Model 'en_core_sci_sm' (0.5.4) was trained with spaCy v3.7.4
and may not be 100% compatible with the current version (3.8.11)
```

### Updated fix_spacy.sh

The [fix_spacy.sh](../fix_spacy.sh) script has been updated with the correct installation order.
