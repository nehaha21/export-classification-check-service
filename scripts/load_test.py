from locust import HttpUser, between, task


class ClassificationLoadUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def classify(self) -> None:
        self.client.post(
            "/classify",
            json={
                "product_description": (
                    "Stainless steel vacuum flask with a moulded "
                    "plastic outer body."
                ),
                "materials": ["stainless steel", "plastic"],
                "additional_information": {
                    "llm_stub": True,
                    "stub_response": {
                        "proposed_classification": "stub-classification",
                        "rule_applied": "stub-GRI",
                        "reasoning": "LLM stub response for load testing.",
                        "citations": ["stub-citation"],
                    },
                },
            },
        )