from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

class PIIService:
    def __init__(self):
        # Initializes the AnalyzerEngine. This will load the spacy en_core_web_lg model by default
        # or en_core_web_sm if configured, depending on Presidio's default behavior, but usually
        # requires an NLP engine configured. Presidio defaults to 'en' with whatever model is available.
        # It's better to explicitly not pass anything if we are okay with the default.
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

    def clean_text(self, text: str) -> str:
        """
        Analyzes the text for PII entities and anonymizes them.
        Uses the default comprehensive list of entities from Presidio.
        """
        if not text:
            return text

        # Analyze the text for PII using default entities
        # By default, language="en" searches for a wide range of PII
        results = self.analyzer.analyze(text=text, language="en")
        
        # Anonymize the found entities
        anonymized_result = self.anonymizer.anonymize(
            text=text, 
            analyzer_results=results
        )
        
        return anonymized_result.text
