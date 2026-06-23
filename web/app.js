const state = {
  questions: [],        // Current phase questions
  allQuestions: [],     // Cumulative: Stage 1 + Stage 2 (for final submit)
  responses: new Map(),
  currentIndex: 0,
  locked: false,
  advanceTimer: null,
  result: null,
  saved: false,
  phase: 1,             // 1 = Stage 1, 2 = Stage 2
  stage1QuestionCount: 0,
  stage2Questions: [],
  stage2Loading: false,
  boundaryBuckets: [],
  stage1Complete: false,
};

const $ = (id) => document.getElementById(id);

const views = {
  setup: $("setupView"),
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
  try {
    const stats = await requestJson("/api/vocabulary/stats");
    const mode = stats.used_fallback ? "内置小词表" : "wordfreq 词库";
    $("bankInfo").textContent = `${mode} · ${stats.size} 个 lemma 词族`;
  } catch (err) {
    $("bankInfo").textContent = "词库状态不可用";
  }
}

async function startTest() {
  setMessage("正在生成测试词表");
  const perBucket = Number($("perBucket").value || 4);
  const data = await fetchQuiz(perBucket);
  state.phase = 1;
  state.questions = data.questions;
  state.allQuestions = [...data.questions];
  state.stage1QuestionCount = data.questions.length;
  state.responses = new Map();
  state.currentIndex = 0;
  state.locked = false;
  state.result = null;
  state.saved = false;
  state.stage1QuestionCount = 0;
  state.stage2Questions = [];
  state.stage2Loading = false;
  state.boundaryBuckets = [];
  state.stage1Complete = false;
  clearAdvanceTimer();
  $("saveStatus").textContent = "";
  $("phaseIndicator").textContent = "第一阶段";
  renderCard();
  showView("test");
  setMessage("");
}

async function fetchQuiz(perBucket) {
  return requestJson(`/api/vocabulary/quiz?per_bucket=${perBucket}&seed=${Date.now() % 100000}`);
}

function renderCard() {
  const total = state.allQuestions.length;
  const answered = state.responses.size;
  const completed = total > 0 && answered === total;

  // Stage 2 loading in progress
  if (state.stage2Loading) {
    $("pageTitle").textContent = "正在准备第二阶段测试...";
    $("wordList").innerHTML = `<p class="status">正在根据第一阶段的回答生成针对性题目...</p>`;
    $("progressText").textContent = "—";
    $("progressBar").value = 50;
    $("submitBtn").classList.add("hidden");
    $("submitBtn").disabled = true;
    return;
  }

  $("progressText").textContent = `${answered}/${total}`;
  $("progressBar").value = total ? (answered / total) * 100 : 0;
  $("submitBtn").classList.toggle("hidden", !completed);
  $("submitBtn").disabled = !completed;

  if (!total) {
    $("pageTitle").textContent = "暂无题目";
    $("wordList").innerHTML = `<p class="status">没有生成测试题</p>`;
    return;
  }

  if (completed) {
    const phase = state.phase === 1 ? "第一阶段" : "第二阶段";
    $("pageTitle").textContent = `${phase}已完成 ${total}/${total} 题`;
    $("wordList").innerHTML = `
      <div class="quizComplete">
        <strong>${phase}全部完成</strong>
        <span>${state.phase === 1 ? "正在分析结果，准备第二阶段测试..." : "提交后会根据所有答对情况估算词汇量。"}</span>
      </div>
    `;
    // Auto-trigger Stage 2 when Stage 1 completes
    if (state.phase === 1 && !state.stage1Complete) {
      state.stage1Complete = true;
      triggerStage2();
    }
    return;
  }

  const current = Math.min(state.currentIndex, total - 1);
  const question = state.questions[current];
  const phaseLabel = state.phase === 2 ? "第二阶段" : "第一阶段";
  $("pageTitle").textContent = `${phaseLabel}：第 ${current + 1}/${total} 题`;

  if (question.mode === "binary" || !Array.isArray(question.options) || question.options.length < 4) {
    renderBinaryCard(question);
    return;
  }

  renderMultipleChoiceCard(question);
}

function renderMultipleChoiceCard(question) {
  const response = state.responses.get(question.word);
  const selected = response?.choice;
  const isAnswered = Boolean(response);
  const isTrap = question.answer === null;

  let feedback;
  if (!isAnswered) {
    feedback = "选择正确中文释义";
  } else if (response.known) {
    feedback = "回答正确";
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
  `;

  const grid = $("wordList").querySelector(".optionGrid");
  question.options.forEach((option, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = optionClass(index, selected, question.answer, isAnswered);
    button.disabled = state.locked;
    button.innerHTML = `<span>${index + 1}</span>${escapeHtml(option)}`;
    button.addEventListener("click", () => chooseOption(index));
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
    btn1.disabled = state.locked;
    btn1.innerHTML = `<span>5</span>以上都不认识`;
    btn1.addEventListener("click", () => chooseUnknownOption(4));
    unknownRow.appendChild(btn1);

    const btn2 = document.createElement("button");
    btn2.type = "button";
    let btn2Class = "unknownBtn";
    if (isAnswered && selected === 5) {
      btn2Class += isTrap ? " selected correct" : " selected wrong";
    }
    btn2.className = btn2Class;
    btn2.disabled = state.locked;
    btn2.innerHTML = `<span>6</span>没有正确答案`;
    btn2.addEventListener("click", () => chooseUnknownOption(5));
    unknownRow.appendChild(btn2);
  }
}

function renderBinaryCard(question) {
  const response = state.responses.get(question.word);
  const selected = response?.known;
  const isAnswered = Boolean(response);
  $("wordList").innerHTML = `
    <article class="quizCard">
      <div class="wordMeta">rank ${escapeHtml(question.rank)} · ${escapeHtml(question.bucket)}</div>
      <div class="quizWord">${escapeHtml(question.word)}</div>
      <div class="binaryHint">该词暂无中文选项，请按原模式作答。</div>
      <div class="binaryGrid">
        <button type="button" class="optionBtn binaryKnown ${isAnswered && selected === true ? "selected correct" : ""}" ${state.locked ? "disabled" : ""}><span>1</span>认识</button>
        <button type="button" class="optionBtn binaryUnknown ${isAnswered && selected === false ? "selected wrong" : ""}" ${state.locked ? "disabled" : ""}><span>2</span>不认识</button>
      </div>
      <p class="feedback">${isAnswered ? "已记录" : "选择你的判断"}</p>
    </article>
  `;
  $("wordList").querySelector(".binaryKnown").addEventListener("click", () => answerBinary(true));
  $("wordList").querySelector(".binaryUnknown").addEventListener("click", () => answerBinary(false));
}

function optionClass(index, selected, answer, isAnswered) {
  const classes = ["optionBtn"];
  if (!isAnswered) return classes.join(" ");
  if (index === answer) classes.push("correct");
  if (index === selected && selected !== answer) classes.push("wrong");
  if (index === selected) classes.push("selected");
  return classes.join(" ");
}

function chooseOption(index) {
  if (state.locked) return;
  const question = state.questions[state.currentIndex];
  if (!question || question.mode === "binary") return;
  const known = index === question.answer;
  state.responses.set(question.word, { word: question.word, known, choice: index });
  state.locked = true;
  renderCard();
  scheduleAdvance();
}

function chooseUnknownOption(index) {
  if (state.locked) return;
  const question = state.questions[state.currentIndex];
  if (!question || question.mode === "binary") return;

  let known;
  if (question.answer === null && index === 5) {
    // Trap question + "没有正确答案" → correct!
    known = true;
  } else {
    known = false;
  }

  state.responses.set(question.word, { word: question.word, known, choice: index });
  state.locked = true;
  renderCard();
  scheduleAdvance();
}

function answerBinary(known) {
  if (state.locked) return;
  const question = state.questions[state.currentIndex];
  if (!question) return;
  state.responses.set(question.word, { word: question.word, known, choice: known ? 0 : 1 });
  state.locked = true;
  renderCard();
  scheduleAdvance();
}

async function triggerStage2() {
  state.stage2Loading = true;
  $("phaseIndicator").textContent = "第二阶段加载中";
  renderCard();
  try {
    const data = await requestJson("/api/vocabulary/quiz-stage2", {
      method: "POST",
      body: JSON.stringify(buildResponseArray()),
    });
    if (data.questions && data.questions.length > 0) {
      state.phase = 2;
      state.questions = data.questions;
      state.allQuestions = state.allQuestions.concat(data.questions);
      state.currentIndex = 0;
      state.boundaryBuckets = data.boundary_buckets || [];
      $("phaseIndicator").textContent = `第二阶段：${data.count} 题`;
    } else {
      // No Stage 2 questions → auto-submit
      $("phaseIndicator").textContent = "测试完成";
      submitTest();
      return;
    }
  } catch (err) {
    setMessage(`第二阶段加载失败: ${err.message}`);
    $("phaseIndicator").textContent = "测试完成";
    // Fall back to submit with Stage 1 only
    submitTest();
    return;
  }
  state.stage2Loading = false;
  renderCard();
}

function buildResponseArray() {
  // Build from allQuestions to include both Stage 1 and Stage 2
  return state.allQuestions.map((item) => ({
    word: item.word,
    known: state.responses.get(item.word)?.known === true,
  }));
}

function scheduleAdvance() {
  clearAdvanceTimer();
  state.advanceTimer = window.setTimeout(() => {
    state.currentIndex += 1;
    state.locked = false;
    state.advanceTimer = null;
    renderCard();
  }, 500);
}

function clearAdvanceTimer() {
  if (state.advanceTimer !== null) {
    window.clearTimeout(state.advanceTimer);
    state.advanceTimer = null;
  }
}

async function submitTest() {
  if (state.responses.size !== state.allQuestions.length) {
    setMessage("请先完成所有题目");
    return;
  }
  clearAdvanceTimer();
  setMessage("正在估算");
  const responses = buildResponseArray();
  const data = await requestJson("/api/estimate", {
    method: "POST",
    body: JSON.stringify({ responses }),
  });
  state.result = data.result;
  // Add stage info to result display
  state.result._stage2_count = state.phase === 2 ? state.allQuestions.length - state.stage1QuestionCount : 0;
  renderResult();
  showView("result");
  setMessage("");
}

function renderResult() {
  const result = state.result;
  const stageNote = result._stage2_count > 0
    ? metric("二阶段补充", `${result._stage2_count} 题`)
    : "";
  const boundaryNote = state.boundaryBuckets.length > 0
    ? metric("重点桶", state.boundaryBuckets.join("、"))
    : "";
  $("resultBox").innerHTML = `
    ${metric("词汇量估计", `${result.point_estimate} 词`)}
    ${metric("原始估计", `${result.raw_estimate} 词`)}
    ${metric("等级", result.level)}
    ${metric("置信度", result.confidence)}
    ${metric("90% 区间", `${result.vocabulary_range[0]} - ${result.vocabulary_range[1]}`)}
    ${metric("有效样本", `${result.sample_size} 题`)}
    ${metric("忽略样本", `${result.ignored_responses} 题`)}
    ${stageNote}
    ${boundaryNote}
  `;
}

function metric(label, value) {
  return `<div class="metric"><span>${escapeHtml(label)}</span><strong>${escapeHtml(String(value))}</strong></div>`;
}

async function saveRecord() {
  if (!state.result) return;
  if (state.saved) {
    $("saveStatus").textContent = "记录已保存";
    return;
  }

  const name = $("studentName").value.trim() || "匿名学生";
  const cetRaw = $("cetScore").value.trim();
  const cetScore = cetRaw === "" ? null : Number(cetRaw);
  const payload = {
    student: { name, cet_score: cetScore },
    responses: buildResponseArray(),
    result: state.result,
  };

  $("saveStatus").textContent = "正在保存";
  const data = await requestJson("/api/tests/save", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  state.saved = true;
  $("saveStatus").textContent = `已保存，记录编号 #${data.record.id}`;
}

// buildResponses is replaced by buildResponseArray (above)

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

$("startBtn").addEventListener("click", () => startTest().catch((err) => setMessage(err.message)));
$("submitBtn").addEventListener("click", () => submitTest().catch((err) => setMessage(err.message)));
$("saveBtn").addEventListener("click", () => saveRecord().catch((err) => ($("saveStatus").textContent = err.message)));
$("restartBtn").addEventListener("click", () => {
  clearAdvanceTimer();
  showView("setup");
});
$("recordsBtn").addEventListener("click", () => loadRecords().catch((err) => setMessage(err.message)));
$("backBtn").addEventListener("click", () => {
  clearAdvanceTimer();
  showView("setup");
});

document.addEventListener("keydown", (event) => {
  if (views.test.classList.contains("hidden") || state.locked) return;
  const key = event.key;
  if (!["1", "2", "3", "4", "5", "6"].includes(key)) return;
  const question = state.questions[state.currentIndex];
  if (!question) return;
  event.preventDefault();
  if (question.mode === "binary") {
    if (key === "1") answerBinary(true);
    if (key === "2") answerBinary(false);
    return;
  }
  if (key === "5") { chooseUnknownOption(4); return; }
  if (key === "6") { chooseUnknownOption(5); return; }
  chooseOption(Number(key) - 1);
});

loadStats();
