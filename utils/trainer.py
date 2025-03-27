from trl import GRPOTrainer

class PrefixGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, prefix_allowed_tokens_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefix_allowed_tokens_fn = prefix_allowed_tokens_fn

    # def _generate(self, input_ids, attention_mask, **kwargs):
    #     kwargs["prefix_allowed_tokens_fn"] = self.prefix_allowed_tokens_fn
    #     return self.model.generate(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         **kwargs
    #     )