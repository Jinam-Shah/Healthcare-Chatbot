import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """
    Attention mechanism for focusing on important words
    """

    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        """
        Args:
            lstm_output: (batch_size, seq_len, hidden_dim)
        Returns:
            context: (batch_size, hidden_dim)
            attention_weights: (batch_size, seq_len)
        """
        # Calculate attention scores
        attention_scores = self.attention(lstm_output)  # (batch, seq_len, 1)
        attention_weights = F.softmax(attention_scores.squeeze(-1), dim=1)  # (batch, seq_len)

        # Apply attention weights
        context = torch.bmm(attention_weights.unsqueeze(1), lstm_output)  # (batch, 1, hidden)
        context = context.squeeze(1)  # (batch, hidden)

        return context, attention_weights


class LSTMAttentionChatbot(nn.Module):
    """
    Advanced chatbot with BiLSTM + Attention mechanism
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes,
                 num_layers=2, dropout=0.3, bidirectional=True,
                 pretrained_embeddings=None, use_attention=True):
        super(LSTMAttentionChatbot, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Calculate LSTM output dimension
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # Attention layer (optional)
        if use_attention:
            self.attention = AttentionLayer(lstm_output_dim)

        # Fully connected layers
        self.fc1 = nn.Linear(lstm_output_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x, lengths=None):
        """
        Args:
            x: (batch_size, seq_len) - word indices
            lengths: (batch_size) - actual sequence lengths (for packing)
        Returns:
            output: (batch_size, num_classes)
            attention_weights: (batch_size, seq_len) if attention enabled
        """
        batch_size = x.size(0)

        # Embedding
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)

        # Pack sequences if lengths provided (for variable-length sequences)
        if lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)

        # Unpack if packed
        if lengths is not None:
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        # Apply attention or use last hidden state
        if self.use_attention:
            context, attention_weights = self.attention(lstm_out)
        else:
            # Use last hidden state
            if self.bidirectional:
                hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
            else:
                hidden = hidden[-1]
            context = hidden
            attention_weights = None

        # Fully connected layers
        out = self.fc1(context)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        if attention_weights is not None:
            return out, attention_weights
        return out

    def predict(self, x, lengths=None):
        """
        Prediction with confidence scores

        Returns:
            predicted_class, confidence, (attention_weights if enabled)
        """
        self.eval()
        with torch.no_grad():
            if self.use_attention:
                output, attention_weights = self.forward(x, lengths)
            else:
                output = self.forward(x, lengths)
                attention_weights = None

            probs = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, dim=1)

            return predicted, confidence, attention_weights


# Backward compatibility with old simple model
class NeuralNet(nn.Module):
    """
    Original simple feedforward network (kept for backward compatibility)
    """

    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out
