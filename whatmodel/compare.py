"""Compare model names from Python: ``whatmodel.compare("a", "b")``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from whatmodel.lookup import DEFAULT_OLLAMA_URL, ModelInfo, explain_many

if TYPE_CHECKING:
    import httpx
    from rich.console import RenderableType


@dataclass(frozen=True)
class ComparedField:
    """One field lined up across the models being compared."""

    field: str
    #: ``True`` for a looked-up fact, ``False`` for something read from the names.
    looked_up: bool
    #: One value per model, in the order they were given. ``None`` where a model has none.
    values: tuple[str | None, ...]

    @property
    def differs(self) -> bool:
        return len(set(self.values)) > 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "source": "lookup" if self.looked_up else "name",
            "values": list(self.values),
            "differs": self.differs,
        }


@dataclass(frozen=True)
class Comparison:
    """Two or more models side by side. ``print()`` it, or read ``fields``."""

    models: tuple[ModelInfo, ...]
    fields: tuple[ComparedField, ...]

    @property
    def differences(self) -> tuple[ComparedField, ...]:
        """Only the fields where the models are not all the same."""
        return tuple(found for found in self.fields if found.differs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "models": [model.to_dict() for model in self.models],
            "fields": [found.to_dict() for found in self.fields],
        }

    def __str__(self) -> str:
        from whatmodel.render import describe_comparison

        return describe_comparison(list(self.models))

    def __rich__(self) -> RenderableType:
        from whatmodel.render import render_compare

        return render_compare(list(self.models))


def compare(
    *names: str,
    offline: bool = False,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    timeout: float | None = None,
    client: httpx.Client | None = None,
    cache_dir: Path | None = None,
) -> Comparison:
    """Explain two or more names and line them up field by field.

    Takes the same options as :func:`whatmodel.explain`. Raises ``ValueError``
    if fewer than two names are given or one of them is not a name at all.
    """
    if len(names) < 2:
        raise ValueError("compare needs at least two names")
    from whatmodel.render import compare_rows

    models = explain_many(
        names,
        offline=offline,
        ollama_url=ollama_url,
        timeout=timeout,
        client=client,
        cache_dir=cache_dir,
    )
    fields = tuple(
        ComparedField(heading, looked_up, tuple(values))
        for heading, looked_up, values in compare_rows(models)
    )
    return Comparison(tuple(models), fields)


__all__ = ["ComparedField", "Comparison", "compare"]
