const state = {
  questions: [],
  responses: new Map(),
  currentIndex: 0,
  phase: "setup",        // setup | quiz_v2 | result | quiz_stage2
  locked: false,
  result: null,
  saved: false,
  stage2Questions: [],
  stage2Responses: new Map(),
  stage2Active: false,
  summaryShown: false,   // true once summary has been displayed
  preRefineResult: null, // snapshot taken before stage2 refinement
  flashChoice: null,
  advanceTimer: null,
  vocabVersion: "v1",
  quizVocabVersion: "v1",
};

const $ = (id) => document.getElementById(id);

const VOCAB_VERSIONS = {
  v1: "原始词库 (v1, 11,418词)",
  v2: "扩展词库 (v2, 19,801词)",
  v2_clusterv1: "扩展词库 (v2, 19,801词)",
};

const views = {
  setup: $("setupView"),
  article: $("articleView"),
  test: $("testView"),
  result: $("resultView"),
  records: $("recordsView"),
};

function showView(name) {
  Object.entries(views).forEach(([key, el]) => {
    el.classList.toggle("hidden", key !== name);
  });
}

function setMessage(text) {
  $("message").textContent = text || "";
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || `请求失败: ${response.status}`);
  }
  return data;
}

async function loadStats() {
  renderVocabInfo();
}

function selectedVocabLabel(version = state.vocabVersion) {
  return VOCAB_VERSIONS[version] || VOCAB_VERSIONS.v1;
}

function renderVocabInfo() {
  $("bankInfo").textContent = `当前词库：${selectedVocabLabel()}`;
}

function withVocabVersion(params = {}, version = state.vocabVersion) {
  return new URLSearchParams({
    ...params,
    vocab_version: version,
  }).toString();
}

// ============================================================
// Phase 1 – Quiz v2（60 道选择题）
// ============================================================

async function startQuizV2() {
  setMessage("正在生成测试题目");
  state.quizVocabVersion = state.vocabVersion;
  try {
    const query = withVocabVersion({
      seed: Date.now() % 100000,
      balanced: "true",
    }, state.quizVocabVersion);
    const data = await requestJson(`/api/vocabulary/quiz-v2?${query}`);
    state.questions = data.questions;
    state.responses = new Map();
    state.currentIndex = 0;
    state.locked = false;
    state.result = null;
    state.saved = false;
    state.stage2Questions = [];
    state.stage2Responses = new Map();
    state.stage2Active = false;
    state.phase = "quiz_v2";
    state.summaryShown = false;
    state.preRefineResult = null;
    $("saveStatus").textContent = "";
    renderQuestion();
    showView("test");
    setMessage("");
  } catch (err) {
    setMessage(`加载题目失败: ${err.message}`);
  }
}

// ============================================================
// Stage 2 – Phase 2 精细化
// ============================================================

async function startStage2() {
  const btn = $("refineBtn");
  btn.disabled = true;
  btn.textContent = "加载中...";
  setMessage("正在准备细化题目");

  try {
    const allResponses = buildResponseArray();
    const data = await requestJson("/api/vocabulary/quiz-v2-stage2", {
      method: "POST",
      body: JSON.stringify({
        responses: allResponses,
        vocab_version: state.quizVocabVersion,
      }),
    });

    if (!data.questions || data.questions.length === 0) {
      setMessage("无需细化");
      btn.disabled = false;
      btn.textContent = "细化不确定的类";
      return;
    }

    // 在精细化前快照当前估算
    state.preRefineResult = state.result ? { ...state.result } : null;

    const qCount = data.questions.length;
    setMessage(`细化 ${qCount} 道不确定类别的题目`);

    state.stage2Questions = data.questions;
    state.stage2Responses = new Map();
    state.stage2Active = true;
    state.currentIndex = 0;
    state.locked = false;
    state.phase = "quiz_stage2";

    renderQuestion();
    showView("test");
  } catch (err) {
    setMessage(`细化加载失败: ${err.message}`);
    btn.disabled = false;
    btn.textContent = "细化不确定的类";
  }
}

// ============================================================
// 渲染当前问题
// ============================================================

function renderQuestion() {
  const isStage2 = state.phase === "quiz_stage2";
  const questions = isStage2 ? state.stage2Questions : state.questions;
  const responses = isStage2 ? state.stage2Responses : state.responses;
  const total = questions.length;

  if (isStage2) {
    $("phaseIndicator").textContent = "细化测试";
    $("phaseIndicator").className = "phase-stage2";
  } else {
    $("phaseIndicator").textContent = "分层测试";
    $("phaseIndicator").className = "phase-main";
  }
  renderProgress(questions, responses);

  if (!total) {
    $("pageTitle").textContent = "暂无题目";
    $("wordList").innerHTML = `<p class="status">没有生成测试题</p>`;
    return;
  }

  const question = questions[state.currentIndex];
  const prefix = isStage2 ? "细化测试" : "测试";
  $("pageTitle").textContent = `${prefix}：第 ${state.currentIndex + 1}/${total} 题`;

  if (question.mode === "binary" || !Array.isArray(question.options) || question.options.length < 4) {
    renderBinaryCard(question, isStage2);
    return;
  }

  renderMCCard(question, isStage2);
}

function renderProgress(questions, responses) {
  const total = questions.length;
  const current = total ? state.currentIndex + 1 : 0;
  $("progressText").textContent = total ? `第 ${current}/${total} 题` : "第 0/0 题";
  $("progressBar").value = current;
  $("progressBar").max = total || 1;

  let dots = $("questionDots");
  if (!dots) {
    dots = document.createElement("div");
    dots.id = "questionDots";
    dots.className = "questionDots";
    $("progressBar").insertAdjacentElement("afterend", dots);
  }

  dots.innerHTML = "";
  questions.forEach((question, index) => {
    const dot = document.createElement("button");
    dot.type = "button";
    dot.className = "questionDot";
    if (responses.has(question.word)) dot.classList.add("answered");
    if (index === state.currentIndex) dot.classList.add("current");
    dot.setAttribute("aria-label", `第 ${index + 1} 题${responses.has(question.word) ? "，已答" : "，未答"}`);
    dot.addEventListener("click", () => {
      clearPendingAdvance();
      state.currentIndex = index;
      renderQuestion();
    });
    dots.appendChild(dot);
  });
}

function renderMCCard(question, isStage2) {
  const questions = isStage2 ? state.stage2Questions : state.questions;
  const responses = isStage2 ? state.stage2Responses : state.responses;
  const response = responses.get(question.word);
  const selected = response?.choice;
  const isAnswered = Boolean(response);
  const isTrap = question.answer === null;

  let feedback;
  if (!isAnswered) {
    feedback = "选择正确中文释义";
  } else if (response.known) {
    feedback = "回答正确 ✅";
  } else if (isTrap) {
    feedback = "回答错误，该题无正确选项";
  } else {
    feedback = `回答错误，正确答案：${question.options[question.answer]}`;
  }

  $("wordList").innerHTML = `
    <article class="quizCard">
      <div class="wordMeta">rank ${escapeHtml(question.rank)} · ${escapeHtml(question.bucket)}</div>
      <div class="quizWord">${escapeHtml(question.word)}</div>
      <div class="optionGrid"></div>
      <div class="unknownRow"></div>
      <p class="feedback ${isAnswered ? (response.known ? "correctText" : "wrongText") : ""}">${escapeHtml(feedback)}</p>
    </article>
    <div class="quizNav">
      <button id="prevBtn" class="ghost" type="button">← 上一题</button>
      <button id="nextBtn" class="primary" type="button">下一题 →</button>
    </div>
  `;

  const grid = $("wordList").querySelector(".optionGrid");
  question.options.forEach((option, i) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = optionClass(i, selected, question.answer, isAnswered);
    if (isFlashChoice(question.word, i)) button.classList.add("answerFlash");
    button.innerHTML = `<span>${i + 1}</span>${escapeHtml(option)}`;
    button.addEventListener("click", () => chooseOption(i, isStage2));
    grid.appendChild(button);
  });

  const unknownRow = $("wordList").querySelector(".unknownRow");
  if (unknownRow) {
    const btn1 = document.createElement("button");
    btn1.type = "button";
    let btn1Class = "unknownBtn";
    if (isAnswered && selected === 4) {
      btn1Class += " selected wrong";
    }
    btn1.className = btn1Class;
    if (isFlashChoice(question.word, 4)) btn1.classList.add("answerFlash");
    btn1.innerHTML = `<span>5</span>以上都不认识`;
    btn1.addEventListener("click", () => chooseUnknown(4, isStage2));
    unknownRow.appendChild(btn1);

    const btn2 = document.createElement("button");
    btn2.type = "button";
    let btn2Class = "unknownBtn";
    if (isAnswered && selected === 5) {
      btn2Class += isTrap ? " selected correct" : " selected wrong";
    }
    btn2.className = btn2Class;
    if (isFlashChoice(question.word, 5)) btn2.classList.add("answerFlash");
    btn2.innerHTML = `<span>6</span>没有正确答案`;
    btn2.addEventListener("click", () => chooseUnknown(5, isStage2));
    unknownRow.appendChild(btn2);
  }

  // 导航按钮可见性
  const prevBtn = $("prevBtn");
  const nextBtn = $("nextBtn");
  const total = questions.length;
  const atFirst = state.currentIndex === 0;
  const atLast = state.currentIndex === total - 1;
  const allAnswered = responses.size === total;
  prevBtn.style.display = atFirst ? "none" : "";
  if (atLast && allAnswered) {
    nextBtn.textContent = "提交估算";
    nextBtn.style.display = "";
  } else if (atLast) {
    nextBtn.style.display = "none";
  } else {
    nextBtn.textContent = "下一题 →";
    nextBtn.style.display = "";
  }
  prevBtn.addEventListener("click", () => goPrev());
  if (atLast && allAnswered) {
    nextBtn.addEventListener("click", () => submitEstimate());
  } else {
    nextBtn.addEventListener("click", () => goNext());
  }
}

function renderBinaryCard(question, isStage2) {
  const questions = isStage2 ? state.stage2Questions : state.questions;
  const responses = isStage2 ? state.stage2Responses : state.responses;
  const response = responses.get(question.word);
  const selected = response?.known;
  const isAnswered = Boolean(response);

  $("wordList").innerHTML = `
    <article class="quizCard">
      <div class="wordMeta">rank ${escapeHtml(question.rank)} · ${escapeHtml(question.bucket)}</div>
      <div class="quizWord">${escapeHtml(question.word)}</div>
      <div class="binaryHint">该词暂无中文选项，请判断是否认识。</div>
      <div class="binaryGrid">
        <button type="button" class="optionBtn binaryKnown ${isAnswered && selected === true ? "selected correct" : ""}"><span>1</span>认识</button>
        <button type="button" class="optionBtn binaryUnknown ${isAnswered && selected === false ? "selected wrong" : ""}"><span>2</span>不认识</button>
      </div>
      <p class="feedback">${isAnswered ? "已记录" : "选择你的判断"}</p>
    </article>
    <div class="quizNav">
      <button id="prevBtn" class="ghost" type="button">← 上一题</button>
      <button id="nextBtn" class="primary" type="button">下一题 →</button>
    </div>
  `;
  if (isFlashChoice(question.word, 0)) {
    $("wordList").querySelector(".binaryKnown").classList.add("answerFlash");
  }
  if (isFlashChoice(question.word, 1)) {
    $("wordList").querySelector(".binaryUnknown").classList.add("answerFlash");
  }
  $("wordList").querySelector(".binaryKnown").addEventListener("click", () => answerBinary(true, isStage2));
  $("wordList").querySelector(".binaryUnknown").addEventListener("click", () => answerBinary(false, isStage2));

  // 导航按钮
  const prevBtn = $("prevBtn");
  const nextBtn = $("nextBtn");
  const total = questions.length;
  const atFirst = state.currentIndex === 0;
  const atLast = state.currentIndex === total - 1;
  const allAnswered = responses.size === total;
  prevBtn.style.display = atFirst ? "none" : "";
  if (atLast && allAnswered) {
    nextBtn.textContent = "提交估算";
    nextBtn.style.display = "";
  } else if (atLast) {
    nextBtn.style.display = "none";
  } else {
    nextBtn.textContent = "下一题 →";
    nextBtn.style.display = "";
  }
  prevBtn.addEventListener("click", () => goPrev());
  if (atLast && allAnswered) {
    nextBtn.textContent = "提交估算";
    nextBtn.addEventListener("click", () => submitEstimate());
  } else {
    nextBtn.addEventListener("click", () => goNext());
  }
}

// ============================================================
// 作答处理器
// ============================================================

function chooseOption(index, isStage2) {
  if (state.locked) return;
  const questions = isStage2 ? state.stage2Questions : state.questions;
  const responses = isStage2 ? state.stage2Responses : state.responses;
  const question = questions[state.currentIndex];
  if (!question || question.mode === "binary") return;
  const known = index === question.answer;
  responses.set(question.word, { word: question.word, known, choice: index });
  flashAndMaybeAdvance(question.word, index, isStage2);
}

function chooseUnknown(index, isStage2) {
  if (state.locked) return;
  const questions = isStage2 ? state.stage2Questions : state.questions;
  const responses = isStage2 ? state.stage2Responses : state.responses;
  const question = questions[state.currentIndex];
  if (!question || question.mode === "binary") return;
  let known;
  if (question.answer === null && index === 5) {
    known = true;
  } else {
    known = false;
  }
  responses.set(question.word, { word: question.word, known, choice: index });
  flashAndMaybeAdvance(question.word, index, isStage2);
}

function answerBinary(known, isStage2) {
  if (state.locked) return;
  const questions = isStage2 ? state.stage2Questions : state.questions;
  const responses = isStage2 ? state.stage2Responses : state.responses;
  const question = questions[state.currentIndex];
  if (!question) return;
  responses.set(question.word, { word: question.word, known, choice: known ? 0 : 1 });
  flashAndMaybeAdvance(question.word, known ? 0 : 1, isStage2);
}

function isFlashChoice(word, choice) {
  return state.flashChoice?.word === word && state.flashChoice?.choice === choice;
}

function clearPendingAdvance() {
  if (state.advanceTimer) {
    clearTimeout(state.advanceTimer);
    state.advanceTimer = null;
  }
  state.locked = false;
  state.flashChoice = null;
}

function flashAndMaybeAdvance(word, choice, isStage2) {
  clearPendingAdvance();
  state.locked = true;
  state.flashChoice = { word, choice };
  renderQuestion();

  state.advanceTimer = setTimeout(() => {
    state.advanceTimer = null;
    state.locked = false;
    state.flashChoice = null;

    const questions = isStage2 ? state.stage2Questions : state.questions;
    const atLast = state.currentIndex >= questions.length - 1;
    if (!atLast) {
      state.currentIndex += 1;
    }
    renderQuestion();
  }, 300);
}

// ============================================================
// 导航
// ============================================================

function goPrev() {
  clearPendingAdvance();
  if (state.currentIndex <= 0) return;
  state.currentIndex -= 1;
  renderQuestion();
}

function goNext() {
  clearPendingAdvance();
  const isStage2 = state.phase === "quiz_stage2";
  const questions = isStage2 ? state.stage2Questions : state.questions;
  const responses = isStage2 ? state.stage2Responses : state.responses;
  const atLast = state.currentIndex >= questions.length - 1;
  if (atLast) {
    if (responses.size === questions.length) {
      submitEstimate();
    }
    return;
  }
  state.currentIndex += 1;
  renderQuestion();
}

// ============================================================
// 总结页（全部测验题答完后显示）
// ============================================================

function showSummary() {
  state.summaryShown = true;
  const responses = state.responses;
  const questions = state.questions;
  const total = questions.length;
  const answered = responses.size;

  let correct = 0;
  let listHtml = "";
  questions.forEach((q, i) => {
    const resp = responses.get(q.word);
    const isAnswered = Boolean(resp);
    const isCorrect = isAnswered && resp.known === true;
    if (isCorrect) correct++;
    const statusIcon = isCorrect ? "✅" : (isAnswered ? "❌" : "⏳");
    const statusClass = isCorrect ? "summary-correct" : (isAnswered ? "summary-wrong" : "summary-unanswered");
    listHtml += `<div class="summary-item ${statusClass}"><span class="summary-idx">${i + 1}</span><span class="summary-word">${escapeHtml(q.word)}</span><span class="summary-status">${statusIcon}</span></div>`;
  });

  $("phaseIndicator").textContent = "测试完成";
  $("phaseIndicator").className = "phase-main";
  state.currentIndex = total ? total - 1 : 0;
  renderProgress(questions, responses);
  $("pageTitle").textContent = `答题汇总：答对 ${correct}/${total}`;

  $("wordList").innerHTML = `
    <div class="summaryContainer">
      <div class="summary-header">
        <div class="summary-score">得分：${correct} / ${total}</div>
        <div class="summary-pct">${total > 0 ? Math.round(correct / total * 100) : 0}%</div>
      </div>
      <div class="summary-list">${listHtml}</div>
      <div class="summary-actions">
        <button id="submitFromSummaryBtn" class="primary" type="button">提交估算</button>
        <button id="modifyFromSummaryBtn" class="ghost" type="button">修改答案</button>
      </div>
    </div>
  `;

  $("submitFromSummaryBtn").addEventListener("click", () => submitEstimate());
  $("modifyFromSummaryBtn").addEventListener("click", () => {
    state.currentIndex = 0;
    renderQuestion();
  });
}

// ============================================================
// 提交估算
// ============================================================

function buildResponseArray() {
  const all = [];
  state.responses.forEach((v) => all.push({ word: v.word, known: v.known === true }));
  state.stage2Responses.forEach((v) => all.push({ word: v.word, known: v.known === true }));
  return all;
}

async function submitEstimate() {
  setMessage("正在估算词汇量");
  const allResponses = buildResponseArray();
  try {
    const data = await requestJson("/api/vocabulary/quiz-v2/estimate", {
      method: "POST",
      body: JSON.stringify({
        responses: allResponses,
        vocab_version: state.quizVocabVersion,
      }),
    });
    state.result = data.result;
    renderResult();
    showView("result");
    setMessage("");
  } catch (err) {
    setMessage(`估算失败: ${err.message}`);
  }
}

function renderResult() {
  const r = state.result;
  const totalQuestions = state.questions.length + state.stage2Questions.length;
  const answered = state.responses.size + state.stage2Responses.size;
  const hint = state.stage2Active ? "含细化测试" : "";

  let comparisonHtml = '';
  if (state.stage2Active && state.preRefineResult) {
    const pre = state.preRefineResult;
    const delta = r.point_estimate - pre.point_estimate;
    const deltaSign = delta >= 0 ? '+' : '';
    const betterClass = delta >= 0 ? 'delta-up' : 'delta-down';
    comparisonHtml = `
      <div class="refineComparison">
        <div class="comparison-title">细化对比</div>
        <div class="comparison-grid">
          <div class="comparison-col">
            <div class="comparison-label">细化前</div>
            <strong>${pre.point_estimate} 词</strong>
            <span>${pre.level || '—'}</span>
          </div>
          <div class="comparison-arrow">→</div>
          <div class="comparison-col">
            <div class="comparison-label">细化后</div>
            <strong>${r.point_estimate} 词</strong>
            <span>${r.level || '—'}</span>
          </div>
          <div class="comparison-delta ${betterClass}">
            <div class="comparison-label">变化</div>
            <strong>${deltaSign}${delta} 词</strong>
            <span>(${r.confidence})</span>
          </div>
        </div>
      </div>
    `;
  }

  $("resultBox").innerHTML = `
    ${comparisonHtml}
    ${metric("词库版本", selectedVocabLabel(state.quizVocabVersion))}
    ${metric("词汇量估计", `${r.point_estimate} 词`)}
    ${metric("θ 能力值", r.theta)}
    ${metric("等级", r.level || "—")}
    ${metric("置信度", r.confidence)}
    ${metric("95% θ 区间", `${r.theta_ci_95[0]} ~ ${r.theta_ci_95[1]}`)}
    ${metric("90% 词汇区间", `${r.vocabulary_range[0]} ~ ${r.vocabulary_range[1]}`)}
    ${metric("有效样本", `${r.sample_size} 题`)}
    ${metric("忽略样本", `${r.ignored_responses} 题`)}
  `;

  const btn = $("refineBtn");
  if (state.stage2Active) {
    btn.classList.add("hidden");
  } else {
    btn.classList.remove("hidden");
    btn.disabled = false;
    btn.textContent = "细化不确定的类";
  }
}

function metric(label, value) {
  return `<div class="metric"><span>${escapeHtml(label)}</span><strong>${escapeHtml(String(value))}</strong></div>`;
}

function optionClass(index, selected, answer, isAnswered) {
  const classes = ["optionBtn"];
  if (!isAnswered) return classes.join(" ");
  if (index === answer) classes.push("correct");
  if (index === selected && selected !== answer) classes.push("wrong");
  if (index === selected) classes.push("selected");
  return classes.join(" ");
}

// ============================================================
// 保存记录
// ============================================================

async function saveRecord() {
  if (!state.result) return;
  if (state.saved) {
    $("saveStatus").textContent = "记录已保存";
    return;
  }

  const name = $("studentName").value.trim() || "匿名学生";
  const payload = {
    student: { name },
    responses: buildResponseArray(),
    result: state.result,
  };

  $("saveStatus").textContent = "正在保存";
  try {
    const data = await requestJson("/api/tests/save", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    state.saved = true;
    $("saveStatus").textContent = `已保存，记录编号 #${data.record.id}`;
  } catch (err) {
    $("saveStatus").textContent = err.message;
  }
}

// ============================================================
// 文章模式（未改动）
// ============================================================

let _articleRequestToken = 0;

async function estimateFromArticle() {
  const text = $("articleInput").value.trim();
  if (!text) {
    setMessage("请先粘贴文章内容");
    return;
  }
  if (text.split(/\s+/).length < 10) {
    setMessage("文章太短，请粘贴至少 10 个词的文本");
    return;
  }

  const btn = $("estimateArticleBtn");
  btn.disabled = true;

  $("articleSaveStatus").textContent = "";
  $("articleResult").innerHTML = "";
  $("articleResult").classList.add("hidden");
  setMessage("正在分析文章...");

  const token = ++_articleRequestToken;

  try {
    const query = withVocabVersion({ _: Date.now() });
    const data = await requestJson(`/api/v2/estimate/article?${query}`, {
      method: "POST",
      body: JSON.stringify({
        article: text,
        vocab_version: state.vocabVersion,
      }),
    });

    if (token !== _articleRequestToken) return;

    const result = data;
    const stats = result.article_stats || {};
    const coverage = result.coverage || {};
    const coveragePercent = Number.isFinite(coverage.stage_vocab)
      ? `${(coverage.stage_vocab * 100).toFixed(1)}%`
      : "—";

    $("articleResult").classList.remove("hidden");
    $("articleResult").innerHTML = `
      ${metric("词库版本", selectedVocabLabel(result.vocab_version || state.vocabVersion))}
      ${metric("词汇量估计", `${result.estimated_vocab ?? "—"} 词`)}
      ${metric("等级", result.level || "—")}
      ${metric("文章难度", result.difficulty_median ?? "—")}
      ${metric("覆盖率", coveragePercent)}
      ${metric(
        "文章总词数",
        `${stats.total_tokens ?? "—"} / 有效词数: ${stats.content_tokens ?? "—"} / 独特词汇: ${stats.unique_content_words ?? "—"}`
      )}
    `;
    setMessage("");
  } catch (err) {
    if (token === _articleRequestToken) {
      $("articleResult").innerHTML = "";
      $("articleResult").classList.add("hidden");
      setMessage(`文章估算失败: ${err.message}`);
    }
  } finally {
    btn.disabled = false;
  }
}

// ============================================================
// 记录
// ============================================================

async function loadRecords() {
  showView("records");
  setMessage("正在加载历史记录");
  const data = await requestJson("/api/tests/records?limit=30");
  $("recordsList").innerHTML = "";
  if (!data.records.length) {
    $("recordsList").innerHTML = `<p class="status">暂无记录</p>`;
  } else {
    data.records.forEach((record) => {
      const el = document.createElement("div");
      el.className = "record";
      el.innerHTML = `
        <strong>${escapeHtml(record.student_name)} · ${record.estimate} 词 · ${escapeHtml(record.level)}</strong>
        <span>${escapeHtml(record.created_at)} · 置信度 ${escapeHtml(record.confidence)} · 区间 ${record.range_low}-${record.range_high}</span>
      `;
      $("recordsList").appendChild(el);
    });
  }
  setMessage("");
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

// ============================================================
// 事件监听器
// ============================================================

$("startBtn").addEventListener("click", () => startQuizV2().catch((err) => setMessage(err.message)));
$("vocabVersionSelect").addEventListener("change", (event) => {
  state.vocabVersion = event.target.value;
  renderVocabInfo();
  $("articleResult").innerHTML = "";
  $("articleResult").classList.add("hidden");
  setMessage("");
});
$("refineBtn").addEventListener("click", () => startStage2().catch((err) => setMessage(err.message)));
$("saveBtn").addEventListener("click", () => saveRecord().catch((err) => ($("saveStatus").textContent = err.message)));
$("restartBtn").addEventListener("click", () => {
  state.phase = "setup";
  state.preRefineResult = null;
  $("refineBtn").classList.remove("hidden");
  showView("setup");
});
$("articleModeBtn").addEventListener("click", () => {
  showView("article");
  $("articleResult").classList.add("hidden");
  $("articleSaveStatus").textContent = "";
  setMessage("");
});
$("backFromArticleBtn").addEventListener("click", () => {
  showView("setup");
  $("articleResult").classList.add("hidden");
  setMessage("");
});
$("estimateArticleBtn").addEventListener("click", () => estimateFromArticle().catch((err) => setMessage(err.message)));
$("recordsBtn").addEventListener("click", () => loadRecords().catch((err) => setMessage(err.message)));
$("backBtn").addEventListener("click", () => {
  showView("setup");
});

document.addEventListener("keydown", (event) => {
  if (views.test.classList.contains("hidden")) return;
  const key = event.key;

  // 导航：左右箭头
  if (key === "ArrowLeft") {
    event.preventDefault();
    goPrev();
    return;
  }
  if (key === "ArrowRight") {
    event.preventDefault();
    goNext();
    return;
  }

  // 答案选择：1-6
  if (!["1", "2", "3", "4", "5", "6"].includes(key)) return;

  const isStage2 = state.phase === "quiz_stage2";
  const questions = isStage2 ? state.stage2Questions : state.questions;
  const question = questions[state.currentIndex];
  if (!question) return;

  event.preventDefault();

  if (question.mode === "binary") {
    if (key === "1") answerBinary(true, isStage2);
    if (key === "2") answerBinary(false, isStage2);
    return;
  }

  if (key === "5") { chooseUnknown(4, isStage2); return; }
  if (key === "6") { chooseUnknown(5, isStage2); return; }
  chooseOption(Number(key) - 1, isStage2);
});

loadStats();
