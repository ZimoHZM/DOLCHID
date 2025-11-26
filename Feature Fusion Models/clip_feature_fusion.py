import torch
import torch.nn as nn


class CLIPFeatureModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        embed_dim: int = 256,
        num_classes: int = 4,
        temperature: float = 0.07,
        lambda_cls: float = 1.0,
    ) -> None:
        """
        Simple dual-encoder model for CBCT/HIST features with
        contrastive alignment + classification heads.
        """
        super().__init__()

        # CBCT encoder + classifier
        self.cbct_fc = nn.Linear(input_dim, embed_dim)
        self.cbct_classifier = nn.Linear(embed_dim, num_classes)

        # HIST encoder + classifier
        self.hist_fc = nn.Linear(input_dim, embed_dim)
        self.hist_classifier = nn.Linear(embed_dim, num_classes)

        # Contrastive learning parameters
        self.temperature = temperature
        self.lambda_cls = lambda_cls  # weight for classification loss

        self.contrastive_loss_fn = nn.CrossEntropyLoss()
        self.classification_loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        cbct_features: torch.Tensor,
        hist_features: torch.Tensor | None = None,
    ):
        """
        Forward pass.

        If `hist_features` is provided (training), compute embeddings and
        classification logits for both modalities.

        If `hist_features` is None (inference), only use CBCT features.
        """
        cbct_features = cbct_features.squeeze(dim=1)
        cbct_embedding = self.cbct_fc(cbct_features)
        cbct_logits = self.cbct_classifier(cbct_embedding)

        if hist_features is not None:
            hist_features = hist_features.squeeze(dim=1)
            hist_embedding = self.hist_fc(hist_features)
            hist_logits = self.hist_classifier(hist_embedding)
            return cbct_embedding, hist_embedding, cbct_logits, hist_logits

        return cbct_embedding, cbct_logits

    def compute_loss(
        self,
        cbct_embeddings: torch.Tensor,
        hist_embeddings: torch.Tensor,
        cbct_logits: torch.Tensor,
        hist_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute total loss = contrastive loss (CBCT ↔ HIST) +
        classification loss (CBCT + HIST).
        """
        # Similarity matrix between CBCT and HIST embeddings
        logits = torch.matmul(cbct_embeddings, hist_embeddings.T) / self.temperature
        labels_contrastive = torch.arange(logits.shape[0], device=logits.device)

        # Symmetric contrastive loss
        contrastive_loss = (
            self.contrastive_loss_fn(logits, labels_contrastive)
            + self.contrastive_loss_fn(logits.T, labels_contrastive)
        )

        # Supervised classification loss for both modalities
        classification_loss = (
            self.classification_loss_fn(cbct_logits, labels)
            + self.classification_loss_fn(hist_logits, labels)
        )

        total_loss = contrastive_loss + self.lambda_cls * classification_loss
        return total_loss
