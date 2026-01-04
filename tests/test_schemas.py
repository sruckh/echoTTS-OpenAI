import unittest

from pydantic import ValidationError

from app.models.schemas import SpeechRequest


class TestSpeechRequestSchema(unittest.TestCase):
    def test_response_format_is_lowercased(self):
        req = SpeechRequest(model="tts-1", input="hello", voice="alloy", response_format="MP3")
        self.assertEqual(req.response_format, "mp3")

    def test_rejects_invalid_response_format(self):
        with self.assertRaises(ValidationError):
            SpeechRequest(model="tts-1", input="hello", voice="alloy", response_format="ogg")

    def test_rejects_speed_out_of_range(self):
        with self.assertRaises(ValidationError):
            SpeechRequest(model="tts-1", input="hello", voice="alloy", speed=10.0)

