"""Versioned desktop inference sidecar."""

from .protocol import AudioPacket, ProtocolError, decode_audio_packet

__all__ = ["AudioPacket", "ProtocolError", "decode_audio_packet"]
