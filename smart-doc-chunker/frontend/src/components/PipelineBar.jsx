const STEPS = [
  { id: "document",   label: "Document"   },
  { id: "chunking",   label: "Chunking"   },
  { id: "embeddings", label: "Embeddings" },
  { id: "retrieval",  label: "Retrieval"  },
  { id: "answer",     label: "Answer"     },
];

export default function PipelineBar({ completedSteps = new Set(), activeStep = null }) {
  return (
    <nav className="pipeline-bar">
      {STEPS.map((step, i) => {
        const done   = completedSteps.has(step.id);
        const active = activeStep === step.id;
        return (
          <div key={step.id} style={{ display: "contents" }}>
            <div className={`ps-step${done ? " done" : active ? " active" : ""}`}>
              <div className="ps-bubble">
                {done ? "✓" : i + 1}
                {active && <span className="ps-ring" />}
              </div>
              <span className="ps-label">{step.label}</span>
            </div>
            {i < STEPS.length - 1 && (
              <div className={`ps-connector${done ? " done" : ""}`} />
            )}
          </div>
        );
      })}
    </nav>
  );
}
