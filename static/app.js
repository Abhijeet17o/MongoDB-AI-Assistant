const chatForm = document.getElementById("chatForm");
const questionInput = document.getElementById("questionInput");
const chatWindow = document.getElementById("chatWindow");
const status = document.getElementById("status");
const prompts = document.querySelectorAll(".prompt");

const setStatus = (text) => {
  status.textContent = text;
};

const addMessage = (role, text) => {
  const message = document.createElement("div");
  message.className = `message ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  message.appendChild(bubble);
  chatWindow.appendChild(message);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return bubble;
};

const formatValue = (value) => {
  if (value === null || value === undefined) return "";
  if (Array.isArray(value)) {
    return value.map((item) => formatValue(item)).filter(Boolean).join(", ");
  }
  if (typeof value === "object") {
    return Object.entries(value)
      .map(([key, entry]) => `${key}: ${formatValue(entry)}`)
      .filter((entry) => entry.trim())
      .join("; ");
  }
  return String(value);
};

const appendDetails = (bubble, sections = []) => {
  if (!Array.isArray(sections) || sections.length === 0) return;
  const details = document.createElement("details");
  details.className = "details";
  const summary = document.createElement("summary");
  summary.className = "details-title";
  summary.textContent = "Details";
  sections.forEach((section, sectionIndex) => {
    const wrapper = document.createElement("div");
    wrapper.className = "details-section";
    const title = document.createElement("div");
    title.className = "details-section-title";
    title.textContent = section.title || `Section ${sectionIndex + 1}`;
    const list = document.createElement("ul");
    list.className = "details-list";

    (section.items || []).forEach((item, index) => {
      const listItem = document.createElement("li");
      listItem.className = "details-item";

      if (item && typeof item === "object" && !Array.isArray(item)) {
        const keys = Object.keys(item);
        if (keys.length === 1 && keys[0] === "value") {
          listItem.textContent = formatValue(item.value);
        } else if (Object.prototype.hasOwnProperty.call(item, "_id") && Object.prototype.hasOwnProperty.call(item, "total")) {
          listItem.textContent = `${formatValue(item._id)}: ${formatValue(item.total)}`;
        } else {
          const heading = document.createElement("div");
          heading.className = "details-heading";
          heading.textContent = `Item ${index + 1}`;
          const sublist = document.createElement("ul");
          sublist.className = "details-sublist";
          keys.forEach((key) => {
            const entry = document.createElement("li");
            entry.textContent = `${key}: ${formatValue(item[key])}`;
            sublist.appendChild(entry);
          });
          listItem.appendChild(heading);
          listItem.appendChild(sublist);
        }
      } else {
        listItem.textContent = formatValue(item);
      }

      list.appendChild(listItem);
    });

    wrapper.appendChild(title);
    wrapper.appendChild(list);
    details.appendChild(wrapper);
  });

  details.appendChild(summary);
  bubble.appendChild(details);
};

let lastQuestion = "";

const sendQuestion = async (question, options = {}) => {
  const { collectionHint = null, showUserMessage = true } = options;
  if (showUserMessage) {
    addMessage("user", question);
  }
  lastQuestion = question;
  setStatus("Thinking...");

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        collection_hint: collectionHint,
      }),
    });

    if (!response.ok) {
      throw new Error(`Request failed with ${response.status}`);
    }

    const data = await response.json();
    const bubble = addMessage("assistant", data.answer || "No response.");
    const results = data.data?.results || [];
    const sections = results.map((result) => ({
      title: result.label || "Results",
      items: result.items || [],
    }));
    if (!sections.length && Array.isArray(data.data?.result?.items)) {
      sections.push({
        title: "Results",
        items: data.data.result.items,
      });
    }
    appendDetails(bubble, sections);
    if (Array.isArray(data.choices) && data.choices.length) {
      const choices = document.createElement("div");
      choices.className = "choice-list";
      data.choices.forEach((choice) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "choice";
        button.textContent = `Use ${choice}`;
        button.addEventListener("click", () => {
          addMessage("user", `Use ${choice}`);
          sendQuestion(lastQuestion, { collectionHint: choice, showUserMessage: false });
        });
        choices.appendChild(button);
      });
      bubble.appendChild(choices);
    }
  } catch (error) {
    addMessage("assistant", `Request failed: ${error.message}`);
  } finally {
    setStatus("Ready");
  }
};

chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;
  questionInput.value = "";
  sendQuestion(question);
});

prompts.forEach((button) => {
  button.addEventListener("click", () => {
    const prompt = button.dataset.prompt;
    if (prompt) {
      sendQuestion(prompt);
    }
  });
});
