let sourcesData = {};

const formatDialogue = (dialogueText) => {
    if (!dialogueText) {
        return "<p>No dialogue available yet. Pose a question to begin.</p>";
    }
    const lines = dialogueText
        .split(/\n+/)
        .filter((line) => line.trim().length > 0);
    return lines
        .map((line) => {
            const withCitations = line.replace(
                /\[cite:(\d+)\]/g,
                (_match, id) => {
                    return `<a href="#" class="citation" data-source-id="${id}">[${id}]</a>`;
                }
            );
            return `<p>${withCitations}</p>`;
        })
        .join("");
};

const updateStatus = (message, isError = false) => {
    const statusEl = document.getElementById("status-message");
    statusEl.textContent = message;
    statusEl.classList.toggle("status-error", isError);
};

const hideModal = () => {
    const modal = document.getElementById("modal");
    modal.classList.add("modal-hidden");
};

const showModalForSource = (sourceId) => {
    const source = sourcesData[sourceId];
    if (!source) {
        updateStatus("Source details could not be found.", true);
        return;
    }
    document.getElementById("source-persona").textContent =
        source.persona || "Unavailable";
    document.getElementById("source-question").textContent =
        source.question_text || "Unavailable";
    document.getElementById("source-verbatim").textContent =
        source.verbatim_text || "Unavailable";
    const modal = document.getElementById("modal");
    modal.classList.remove("modal-hidden");
};

const bootstrap = () => {
    const form = document.getElementById("question-form");
    const questionInput = document.getElementById("question-input");
    const submitButton = form.querySelector("button[type='submit']");
    const dialogueContainer = document.getElementById("dialogue-container");
    const modal = document.getElementById("modal");
    const modalClose = document.getElementById("close-modal");

    form.addEventListener("submit", async (event) => {
        event.preventDefault();
        const question = questionInput.value.trim();
        if (!question) {
            updateStatus("Please enter a question before submitting.", true);
            return;
        }
        updateStatus("Generating Socratic dialogue...", false);
        dialogueContainer.innerHTML = "";
        sourcesData = {};
        submitButton.disabled = true;
        submitButton.textContent = "Thinking...";
        try {
            const response = await fetch("/generate_dialogue", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question }),
            });
            const payload = await response.json();
            if (!response.ok) {
                throw new Error(
                    payload.error || "An unexpected error occurred."
                );
            }
            sourcesData = payload.sources || {};
            dialogueContainer.innerHTML = formatDialogue(payload.dialogue);
            updateStatus(
                "Dialogue ready. Click a citation to inspect the source."
            );
        } catch (error) {
            updateStatus(error.message, true);
        } finally {
            submitButton.disabled = false;
            submitButton.textContent = "Generate Dialogue";
        }
    });

    dialogueContainer.addEventListener("click", (event) => {
        const citationLink = event.target.closest(".citation");
        if (!citationLink) {
            return;
        }
        event.preventDefault();
        const sourceId = citationLink.dataset.sourceId;
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
};

document.addEventListener("DOMContentLoaded", bootstrap);
