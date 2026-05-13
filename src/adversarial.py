import torch


class FGM:
    def __init__(self, model, epsilon=1.0, emb_name="word_embeddings"):
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}

    def enable_attack_grad(self):
        matched = False
        for name, param in self.model.named_parameters():
            if self.emb_name in name:
                matched = True
                param.requires_grad_(True)

        if not matched:
            raise ValueError(f"No parameter name contains emb_name={self.emb_name!r}.")

    def attack(self):
        attacked = 0
        self.backup = {}

        for name, param in self.model.named_parameters():
            if self.emb_name not in name:
                continue
            if not param.requires_grad or param.grad is None:
                continue

            norm = torch.norm(param.grad)
            if not torch.isfinite(norm) or norm.item() == 0:
                continue

            self.backup[name] = param.data.clone()
            param.data.add_(self.epsilon * param.grad / norm)
            attacked += 1

        return attacked

    def restore(self):
        if not self.backup:
            return

        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])

        self.backup = {}
