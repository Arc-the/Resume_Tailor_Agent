"""Runtime configuration for the resume tailoring pipeline."""

from dataclasses import dataclass, field

from langchain_core.language_models.chat_models import BaseChatModel


@dataclass
class PipelineConfig:
    # LLM settings
    provider: str = "openai"  # "openai" or "gemini"
    model_name: str = "gpt-4o"
    temperature: float = 0.2
    generation_temperature: float = 0.4  # slightly higher for resume writing

    # Evaluation thresholds
    min_passing_score: float = 7.0
    max_iterations: int = 3

    # Content selection
    max_experiences: int = 4  # max experience blocks to keep (0 = no limit)
    min_bullets_per_block: int = 2  # never strip a kept block below this
    base_bullet_target: float = 2.5  # baseline bullet count per block
    max_bullet_target: int = 5  # cap for highest-scoring blocks

    # Validation settings
    bullet_match_threshold: float = 0.85  # SequenceMatcher ratio for fuzzy matching

    # Research settings
    enable_research: bool = False
    max_search_results: int = 5

    # Constraints defaults
    default_constraints: dict = field(default_factory=lambda: {
        "max_pages": 1,
        "tone": "professional",
    })

    def get_llm(self, temperature: float | None = None) -> BaseChatModel:
        """Create an LLM instance based on the configured provider."""
        temp = temperature if temperature is not None else self.temperature

        if self.provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=temp,
            )

        # Default: OpenAI
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=self.model_name,
            temperature=temp,
        )


# Module-level default config, overridable before pipeline runs
_config = PipelineConfig()


def get_config() -> PipelineConfig:
    return _config


def set_config(config: PipelineConfig):
    global _config
    _config = config
