from dataclasses import dataclass

@dataclass
class Chunk:
    chunk_id: str
    path: str
    language: str
    symbol: str
    chunk_type: str
    start_line: int
    end_line: int
    code: str
    docstring: str | None

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "path": self.path,
            "language": self.language,
            "symbol": self.symbol,
            "chunk_type": self.chunk_type,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "code": self.code,
            "docstring": self.docstring,
        }