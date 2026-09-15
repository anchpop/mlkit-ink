#!/usr/bin/env python3
"""Resolve BCP-47 identifiers against ML Kit's shipped pack catalog.

Exact identifiers win. Fallback keeps the language and private-use variant,
then prefers matching script/region and the least-specific catalog entry.
Unknown languages never silently become English; gestures never become text.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TypedDict

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

try:
    from . import fetch_pack
except ImportError:  # direct CLI invocation
    import fetch_pack

ROOT = Path(__file__).resolve().parent.parent


class PackNames(TypedDict):
    recospec: str
    tflite: str
    fst: str | None


class PackPaths(TypedDict):
    recospec: Path
    tflite: Path
    fst: Path | None


def _mapping_message():
    """The small, independently known packmapping.pb schema (SPEC section 1)."""
    schema = descriptor_pb2.FileDescriptorProto(name="packmapping.proto", syntax="proto2")
    entry = schema.message_type.add(name="PackEntry")
    for number, name in ((1, "tag"), (5, "recospec"), (6, "tflite"), (7, "fst")):
        entry.field.add(name=name, number=number, type=9, label=1)
    mapping = schema.message_type.add(name="PackMapping")
    mapping.field.add(name="entries", number=1, type=11, label=3, type_name=".PackEntry")
    pool = descriptor_pool.DescriptorPool()
    pool.Add(schema)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName("PackMapping"))


def _parts(tag: str) -> tuple[str, str | None, str | None, str]:
    parts = tag.replace("_", "-").lower().split("-")
    private = parts.index("x") if "x" in parts else len(parts)
    core = parts[:private]
    script = next((p for p in core[1:] if len(p) == 4 and p.isalpha()), None)
    region = next((p for p in core[1:] if (len(p) == 2 and p.isalpha())
                   or (len(p) == 3 and p.isdigit())), None)
    return core[0], script, region, "-".join(parts[private:])


class PackResolver:
    def __init__(self, mapping: Path = ROOT / "packmapping.pb",
                 manifest: Path = ROOT / "manifest.json"):
        self.manifest = {p["name"]: p for p in json.loads(Path(manifest).read_text())["packs"]}
        message = _mapping_message().FromString(Path(mapping).read_bytes())
        self.entries: dict[str, PackNames] = {}
        for entry in message.entries:
            names = PackNames(recospec=entry.recospec, tflite=entry.tflite,
                              fst=entry.fst if entry.HasField("fst") else None)
            if not entry.tag or not names["recospec"] or not names["tflite"]:
                raise ValueError("Incomplete pack mapping entry")
            for name in names.values():
                if name is not None and name not in self.manifest:
                    raise ValueError(f"Mapping references missing pack {name!r}")
            if entry.tag in self.entries:
                raise ValueError(f"Duplicate language tag {entry.tag!r}")
            self.entries[entry.tag] = names
        self._tags = {tag.lower(): tag for tag in self.entries}

    @property
    def languages(self) -> tuple[str, ...]:
        return tuple(sorted(self.entries))

    def resolve_tag(self, language_tag: str) -> str:
        normalized = language_tag.strip().replace("_", "-").lower()
        if normalized in self._tags:
            return self._tags[normalized]
        language, script, region, private = _parts(normalized)
        candidates = []
        for tag in self.entries:
            lang, scr, reg, variant = _parts(tag)
            if lang != language or variant != private:
                continue
            # Never substitute a known, conflicting script. Hani is the
            # catalog's umbrella for the standard Hans/Hant identifiers.
            if script and scr and script != scr and not (script in ("hans", "hant") and scr == "hani"):
                continue
            preferred_region = region or ({"hans": "cn", "hant": "tw"}.get(script))
            score = (int(bool(preferred_region) and reg == preferred_region),
                     int(bool(script) and scr == script),
                     -len(tag.split("-")), tag)
            candidates.append((score, tag))
        if not candidates:
            raise KeyError(f"No Digital Ink model for {language_tag!r}")
        return max(candidates)[1]

    def resolve(self, language_tag: str) -> PackNames:
        return self.entries[self.resolve_tag(language_tag)].copy()

    def ensure(self, language_tag: str, dest: Path = ROOT / "models") -> PackPaths:
        result = {}
        for kind, name in self.resolve(language_tag).items():
            if name is None:
                result[kind] = None
                continue
            # Isolate archives: two language packs can contain the same basename.
            paths = fetch_pack.fetch(name, Path(dest) / name)
            suffix = {"recospec": ".recospec.local", "tflite": ".tflite", "fst": ".compact.fst.local"}[kind]
            files = [p for p in paths if p.is_file() and p.name.endswith(suffix)]
            if not files:  # Some FST pack generations use a different filename.
                files = [p for p in paths if p.is_file()]
            if len(files) != 1:
                raise ValueError(f"Expected one {kind} artifact in {name}, got {files}")
            result[kind] = files[0]
        return PackPaths(**result)

    def counts(self) -> dict[str, int]:
        nets = {entry["tflite"] for entry in self.entries.values()}
        # Net family names include scripts omitted from short BCP-47 tags.
        scripts = {name.removeprefix("indy_lstm_").removeprefix("scribe_").removeprefix("lstm_").split("_")[0]
                   for name in nets}
        return {"language_tags": len(self.entries),
                "languages": len({_parts(tag)[0] for tag in self.entries}),
                "explicit_script_subtags": len({_parts(tag)[1] for tag in self.entries} - {None}),
                "script_families": len(scripts - {"autodraw", "emoji", "shapes"}),
                "net_families": len(scripts), "nets": len(nets),
                "text_nets": len({e["tflite"] for e in self.entries.values() if e["fst"]}),
                "recospecs": len({e["recospec"] for e in self.entries.values()})}


_default: PackResolver | None = None


def _resolver() -> PackResolver:
    global _default
    if _default is None:
        _default = PackResolver()
    return _default


def resolve(language_tag: str) -> PackNames:
    return _resolver().resolve(language_tag)


def ensure(language_tag: str, dest: Path = ROOT / "models") -> PackPaths:
    return _resolver().ensure(language_tag, dest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("language_tag", nargs="?")
    parser.add_argument("--list-languages", action="store_true")
    args = parser.parse_args()
    resolver = _resolver()
    if args.list_languages:
        print("\n".join(resolver.languages))
        print(json.dumps(resolver.counts(), sort_keys=True))
    elif args.language_tag:
        print(json.dumps(ensure(args.language_tag), default=str, indent=2))
    else:
        parser.error("provide a language tag or --list-languages")


if __name__ == "__main__":
    main()
