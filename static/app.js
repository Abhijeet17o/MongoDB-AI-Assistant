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
