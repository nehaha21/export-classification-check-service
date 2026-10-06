from locust import HttpUser, between, task


class CachedClassificationLoadUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def classify_cached_request(self) -> None:
        self.client.post(
            "/classify",
            json={
                "product_description": "Cache load test product",
                "materials": ["steel"],
                "additional_information": {
                    "llm_stub": True,
                },
            },
        )