let sourcesData = {};
let conversationHistory = [];
let isSending = false;
let exchangeCounter = 0;

const HISTORY_SLICE = 6;

const chatArea = document.getElementById("chat-area");
const welcomeState = document.getElementById("welcome-state");
const messageInput = document.getElementById("message-input");
const sendBtn = document.getElementById("send-btn");
const newChatBtn = document.getElementById("new-chat-btn");
const modal = document.getElementById("modal");
const modalClose = document.getElementById("close-modal");

const autoResizeTextarea = (textarea) => {
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 150)}px`;
};

const scrollToBottom = () => {
    chatArea.scrollTop = chatArea.scrollHeight;
};

const parseDialogue = (dialogueText) => {
    if (!dialogueText) {
        console.warn("PlatoAI debug | empty dialogueText received");
        return [];
    }

    const lines = dialogueText
        .replace(/\*\*(Socrates|Glaucon):\*\*/gi, (match, name) => `${name}:`)
        .replace(/\*\*(Socrates|Glaucon)\*\*:?/gi, (match, name) => `${name}:`)
        .replace(/\*(Socrates|Glaucon):\*/gi, (match, name) => `${name}:`)
        .replace(/\*(Socrates|Glaucon)\*:?/gi, (match, name) => `${name}:`)
        .replace(
            /^(Socrates|Glaucon)\s*[–—-]\s*/gim,
            (match, name) => `${name}: `
        )
        .replace(/\u201c|\u201d/g, '"')
        .split(/\n+/)
        .filter((line) => line.trim().length);
    console.log("PlatoAI debug | normalized dialogue lines:", lines);
    const messages = [];
    let currentSpeaker = null;
    let currentText = "";

    lines.forEach((line) => {
        const normalizedLine = line.replace(
            /^["'](Socrates|Glaucon):/i,
            (match, name) => `${name}:`
        );
        const match = normalizedLine.match(/^(Socrates|Glaucon):/);
        console.log(
            "PlatoAI debug | examining line:",
            normalizedLine,
            "match:",
            match
        );
        if (match) {
            if (currentSpeaker && currentText) {
                messages.push({
                    speaker: currentSpeaker,
                    text: currentText.trim(),
                });
                console.log(
                    "PlatoAI debug | pushed message: ",
                    currentSpeaker,
                    currentText.trim()
                );
            }
            currentSpeaker = match[1];
            currentText = normalizedLine
                .replace(/^(Socrates|Glaucon):/, "")
                .trim();
            console.log(
                "PlatoAI debug | new speaker ->",
                currentSpeaker,
                "text:",
                currentText
            );
        } else {
            currentText = `${currentText} ${normalizedLine.trim()}`.trim();
            console.log("PlatoAI debug | appended text ->", currentText);
        }
    });

    if (currentSpeaker && currentText) {
        messages.push({ speaker: currentSpeaker, text: currentText.trim() });
        console.log(
            "PlatoAI debug | final push:",
            currentSpeaker,
            currentText.trim()
        );
    }

    return messages;
};

const formatWithCitations = (text, prefix) =>
    text.replace(/\[cite:(\d+)\]/g, (_match, id) => {
        return `<a href="#" class="citation" data-source-id="${prefix}-${id}">[${id}]</a>`;
    });

const ensureChatStarted = () => {
    if (welcomeState.style.display !== "none") {
        welcomeState.style.display = "none";
    }
};

const addUserMessage = (text) => {
    ensureChatStarted();
    const wrapper = document.createElement("div");
    wrapper.className = "message-group";
    wrapper.innerHTML = `<div class="user-message">${text}</div>`;
    chatArea.appendChild(wrapper);
    scrollToBottom();
};

const addThinkingIndicator = () => {
    const thinking = document.createElement("div");
    thinking.className = "thinking";
    thinking.id = "thinking-indicator";
    thinking.innerHTML = `
        <div class="avatar socrates">${"\u03a3"}</div>
        <div class="message-content">
            <div class="thinking-bubble">
                <div class="thinking-dots">
                    <div class="thinking-dot"></div>
                    <div class="thinking-dot"></div>
                    <div class="thinking-dot"></div>
                </div>
            </div>
        </div>
    `;
    chatArea.appendChild(thinking);
    scrollToBottom();
};

const removeThinkingIndicator = () => {
    const thinking = document.getElementById("thinking-indicator");
    if (thinking) {
        thinking.remove();
    }
};

const addDialogueMessages = (messages, prefix) => {
    removeThinkingIndicator();
    messages.forEach((message, index) => {
        setTimeout(() => {
            const wrapper = document.createElement("div");
            wrapper.className = "dialogue-message";
            const speaker = message.speaker.toLowerCase();
            const avatarLetter = speaker === "socrates" ? "\u03a3" : "\u0393";
            const formatted = formatWithCitations(message.text, prefix);
            wrapper.innerHTML = `
                <div class="avatar ${speaker}">${avatarLetter}</div>
                <div class="message-content">
                    <div class="speaker-name ${speaker}">${message.speaker}</div>
                    <div class="message-bubble">${formatted}</div>
                </div>
            `;
            chatArea.appendChild(wrapper);
            scrollToBottom();
        }, index * 300);
    });
};

const showErrorMessage = (errorMessage) => {
    removeThinkingIndicator();
    const wrapper = document.createElement("div");
    wrapper.className = "message-group";
    wrapper.innerHTML = `
        <div class="dialogue-message">
            <div class="avatar socrates">${"\u03a3"}</div>
            <div class="message-content">
                <div class="message-bubble" style="color: #ff6b9d;">
                    I encountered an error: ${errorMessage}
                </div>
            </div>
        </div>
    `;
    chatArea.appendChild(wrapper);
    scrollToBottom();
};

const showModalForSource = (sourceId) => {
    const source = sourcesData[sourceId];
    if (!source) {
        showErrorMessage("Source details could not be found.");
        return;
    }
    document.getElementById("source-name").textContent =
        source.respondent_name || source.persona || "Unavailable";
    document.getElementById("source-persona").textContent =
        source.persona || "Unavailable";
    document.getElementById("source-question").textContent =
        source.full_question || source.question_text || "Unavailable";
    document.getElementById("source-verbatim").textContent =
        source.verbatim_text || "Unavailable";
    modal.classList.remove("modal-hidden");
};

const hideModal = () => {
    modal.classList.add("modal-hidden");
};

const startNewChat = () => {
    conversationHistory = [];
    sourcesData = {};
    exchangeCounter = 0;
    chatArea.innerHTML = "";
    chatArea.appendChild(welcomeState);
    welcomeState.style.display = "block";
    messageInput.value = "";
    autoResizeTextarea(messageInput);
    sendBtn.disabled = true;
};

const sendMessage = async (question) => {
    const trimmed = question.trim();
    if (!trimmed || isSending) {
        return;
    }
    isSending = true;
    addUserMessage(trimmed);
    messageInput.value = "";
    autoResizeTextarea(messageInput);
    sendBtn.disabled = true;
    addThinkingIndicator();

    try {
        const response = await fetch("/generate_dialogue", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                question: trimmed,
                conversation_history: conversationHistory.slice(-HISTORY_SLICE),
            }),
        });

        const payload = await response.json();
        console.log("PlatoAI debug | raw dialogue payload:", payload.dialogue);
        if (!response.ok) {
            throw new Error(payload.error || "An unexpected error occurred.");
        }

        const sourcePrefix = `exchange-${exchangeCounter}`;
        Object.entries(payload.sources || {}).forEach(([id, source]) => {
            sourcesData[`${sourcePrefix}-${id}`] = source;
        });

        conversationHistory.push({ role: "user", content: trimmed });
        conversationHistory.push({
            role: "assistant",
            content: payload.dialogue,
        });

        const messages = parseDialogue(payload.dialogue);
        console.log("PlatoAI debug | parsed messages:", messages);
        addDialogueMessages(messages, sourcePrefix);
        exchangeCounter += 1;
    } catch (error) {
        showErrorMessage(error.message);
    } finally {
        isSending = false;
        sendBtn.disabled = messageInput.value.trim().length === 0;
    }
};

messageInput.addEventListener("input", () => {
    autoResizeTextarea(messageInput);
    sendBtn.disabled = messageInput.value.trim().length === 0 || isSending;
});

messageInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage(messageInput.value);
    }
});

sendBtn.addEventListener("click", () => sendMessage(messageInput.value));

newChatBtn.addEventListener("click", startNewChat);

document.querySelectorAll(".suggestion-card").forEach((card) => {
    card.addEventListener("click", () => {
        const question = card.dataset.question;
        sendMessage(question);
    });
});

chatArea.addEventListener("click", (event) => {
    const citation = event.target.closest(".citation");
    if (!citation) {
        return;
    }
    event.preventDefault();
    const sourceId = citation.dataset.sourceId;
    showModalForSource(sourceId);
});

modal.addEventListener("click", (event) => {
    if (event.target === modal) {
        hideModal();
    }
});

modalClose.addEventListener("click", hideModal);

window.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
        hideModal();
    }
});

document.addEventListener("DOMContentLoaded", () => {
    autoResizeTextarea(messageInput);
});
