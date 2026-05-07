"""Schema for upd_hypo step — hypothesis revision and ledger.

Two relation typologies are emitted by upd_hypo:
- H↔H (gap between adjacent hypotheses): Moulines structuralist typology
  (evolution / embedding / replacement).
- A↔A (each new artifact's edges to its in_dependencies): MultiCite citation
  function typology (Lauscher, Ko, Kuehl, Johnson, Cohan, Jurgens & Lo, NAACL 2022)
  reduced to 6 plain-English types: background / motivation / uses / extends /
  similarities / differences (FUTURE_WORK dropped — every artifact in the trace
  is already realized).
"""

from typing import Annotated, Literal

from aii_lib.prompts import LLMPrompt, LLMPromptModel, LLMStructOut, LLMStructOutModel
from pydantic import Field


class ArtifactRelation(LLMPromptModel, LLMStructOutModel):
    """One typed A↔A edge between a dependent artifact and one of its in_dependencies.

    MultiCite citation-function typology (Lauscher et al., NAACL 2022),
    reduced to 6 plain-English types.
    """

    from_id: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="ID of the predecessor artifact (the one being depended on)"
    )
    to_id: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="ID of the dependent artifact (the new artifact this iteration)"
    )
    relation_type: Annotated[
        Literal["background", "motivation", "uses", "extends", "similarities", "differences"],
        LLMPrompt,
        LLMStructOut,
    ] = Field(
        description=(
            "MultiCite citation-function type for the predecessor→dependent edge: "
            "'background' — predecessor is treated as background context; "
            "'motivation' — predecessor motivated this artifact's research; "
            "'uses' — this artifact uses the predecessor's data, method, or output; "
            "'extends' — this artifact extends the predecessor; "
            "'similarities' — this artifact's results agree with the predecessor's; "
            "'differences' — this artifact's results disagree with the predecessor's."
        )
    )
    relation_rationale: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        max_length=50,
        description="Brief rationale for this relation type (max 50 characters).",
    )


class RevisedHypothesis(LLMPromptModel, LLMStructOutModel):
    """Revised hypothesis after reviewing iteration results.

    Output matches the hypothesis dict structure so it can replace the
    original hypothesis in subsequent iterations.
    """

    kind: Literal["revised_hypothesis"] = "revised_hypothesis"
    title: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="Revised hypothesis title (may be unchanged if still accurate)"
    )
    hypothesis: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="Revised hypothesis statement — what we now believe based on evidence"
    )
    description: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="Revised description of the hypothesis and its scope"
    )
    relation_rationale: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        max_length=50,
        description="Brief rationale for the H↔H revision type (max 50 characters).",
    )
    confidence_delta: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="How confidence changed: 'increased', 'decreased', or 'unchanged'"
    )
    key_changes: Annotated[list[str], LLMPrompt, LLMStructOut] = Field(
        description="Bullet list of specific changes made to the hypothesis"
    )
    relation_type: Annotated[
        Literal["evolution", "embedding", "replacement"], LLMPrompt, LLMStructOut
    ] = Field(
        description=(
            "Moulines's structuralist typology of this hypothesis revision: "
            "'evolution' — refining specialised claims while keeping the same conceptual frame; "
            "'embedding' — the previous hypothesis is now a special case of a broader frame; "
            "'replacement' — rejecting the previous frame entirely (incommensurable, Kuhnian revolution)."
        )
    )
    artifact_relations: Annotated[list[ArtifactRelation], LLMPrompt, LLMStructOut] = Field(
        default_factory=list,
        description=(
            "Typed A↔A edges for this iteration's new artifacts. Emit one entry "
            "per (predecessor → dependent) edge for every in_dependency on each "
            "artifact produced this iteration."
        ),
    )

    def to_hypothesis_dict(self, original: dict) -> dict:
        """Merge revised fields into the original hypothesis dict.

        Preserves metadata fields (hypothesis_id, is_seeded, model, seeds)
        while updating content fields.
        """
        updated = dict(original)
        updated["title"] = self.title
        updated["hypothesis"] = self.hypothesis
        updated["_relation_rationale"] = self.relation_rationale
        updated["_confidence_delta"] = self.confidence_delta
        updated["_key_changes"] = self.key_changes
        updated["relation_type"] = self.relation_type
        return updated
