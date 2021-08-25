import torch.nn as nn


class MetaModel(nn.Module):
    def __init__(self, ae, classifier, name='network'):
        super(MetaModel, self).__init__()

        self.encoder = ae.encoder
        self.classifier = classifier
        self.name = name

    def forward(self, x):
        x_enc = self.encoder(x).squeeze()
        x_out = self.classifier(x_enc)
        return x_out.squeeze(-1)

    def get_name(self):
        return self.name


class AEandClass(MetaModel):
    def __init__(self, ae, **kwargs):
        super(AEandClass, self).__init__(ae, **kwargs)
        self.decoder = ae.decoder

    def forward(self, x):
        x_enc = self.encoder(x)
        xhat = self.decoder(x_enc)
        x_out = self.classifier(x_enc.squeeze(-1))
        return xhat, x_out, x_enc


class NonLinClassifier(nn.Module):
    def __init__(self, d_in, n_class, d_hidd=16, activation=nn.ReLU(), dropout=0.1, norm='batch'):
        """
        norm : str : 'batch' 'layer' or None
        """
        super(NonLinClassifier, self).__init__()

        self.dense1 = nn.Linear(d_in, d_hidd)

        if norm == 'batch':
            self.norm = nn.BatchNorm1d(d_hidd)
        elif norm == 'layer':
            self.norm = nn.LayerNorm(d_hidd)
        else:
            self.norm = None

        self.act = activation
        self.dropout = nn.Dropout(dropout)
        self.dense2 = nn.Linear(d_hidd, n_class)

        self.layers = [self.dense1, self.norm, self.act, self.dropout, self.dense2]
        self.net = nn.Sequential(*[x for x in self.layers if x is not None])

    def forward(self, x):
        out = self.net(x)
        return out


class LinClassifier(nn.Module):
    def __init__(self, d_in, n_class):
        super(LinClassifier, self).__init__()
        self.dense = nn.Linear(d_in, n_class)

    def forward(self, x):
        out = self.dense(x)
        return out
