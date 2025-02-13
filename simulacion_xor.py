import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class XOR(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=2, output_dim=1):
        super(XOR, self).__init__()
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.sigmoid(self.lin1(x))
        x = torch.sigmoid(self.lin2(x))
        return x

def train_model(model, X, Y, epochs=3000, lr=0.02):
    loss_func = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_hat = model(X)
        loss = loss_func(y_hat, Y)
        loss.backward()
        optimizer.step()
    return model

def draw_network(screen, model, font, inputs=None):
    neuron_color = (0, 0, 255)
    active_color = (255, 0, 0)
    inactive_color = (0, 0, 255)
    radius = 20

    input_neurons = [(100, 150), (100, 250)]
    hidden_neurons = [(300, 100), (300, 200)]
    output_neuron = (500, 150)

    weights1 = model.lin1.weight.data.numpy()
    weights2 = model.lin2.weight.data.numpy()

    for i, input_pos in enumerate(input_neurons):
        for j, hidden_pos in enumerate(hidden_neurons):
            weight = weights1[j, i]
            color = active_color if weight > 0 else inactive_color
            pygame.draw.line(screen, color, input_pos, hidden_pos, int(abs(weight * 10)))

    for j, hidden_pos in enumerate(hidden_neurons):
        weight = weights2[0, j]
        color = active_color if weight > 0 else inactive_color
        pygame.draw.line(screen, color, hidden_pos, output_neuron, int(abs(weight * 10)))

    for idx, pos in enumerate(input_neurons + hidden_neurons + [output_neuron]):
        color = neuron_color
        if inputs is not None:
            if idx < 2 and inputs[idx] == 1:
                color = active_color
            elif idx == 4 and inputs[-1] > 0.5:
                color = active_color
        pygame.draw.circle(screen, color, pos, radius)

    screen.blit(font.render("Entrada 1", True, (0, 0, 0)), (20, 140))
    screen.blit(font.render("Entrada 2", True, (0, 0, 0)), (20, 240))
    screen.blit(font.render("Capa Oculta", True, (0, 0, 0)), (250, 50))
    screen.blit(font.render("Salida", True, (0, 0, 0)), (520, 130))

    if inputs is not None:
        screen.blit(font.render(f"({inputs[0]}, {inputs[1]})", True, (0, 0, 0)), (20, 100))

def main():
    pygame.init()
    screen = pygame.display.set_mode((600, 500))
    pygame.display.set_caption("Red Neuronal XOR con Pygame")
    font = pygame.font.Font(None, 36)

    calc_text = font.render("Calcular", True, (0, 0, 0), (200, 200, 200))
    calc_rect = calc_text.get_rect(center=(200, 450))

    test_text = font.render("Probar", True, (0, 0, 0), (200, 200, 200))
    test_rect = test_text.get_rect(center=(400, 450))

    X = torch.Tensor([[0,0],[0,1],[1,0],[1,1]])
    Y = torch.Tensor([[0],[1],[1],[0]])
    model = XOR()

    running = True
    show_network = False
    inputs = None
    calc_pressed = False  # Variable para ocultar el botón "Calcular"

    while running:
        screen.fill((255, 255, 255))

        if not calc_pressed:
            screen.blit(calc_text, calc_rect)  # Solo muestra el botón "Calcular" si no se ha presionado

        screen.blit(test_text, test_rect)  # El botón "Probar" siempre es visible

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if not calc_pressed and calc_rect.collidepoint(event.pos):
                    model = train_model(model, X, Y)
                    show_network = True
                    calc_pressed = True  # Oculta el botón "Calcular" tras la primera pulsación
                elif test_rect.collidepoint(event.pos) and show_network:
                    idx = np.random.randint(0, 4)
                    sample = X[idx]
                    output = model(sample)
                    inputs = (int(sample[0].item()), int(sample[1].item()), output.item())

        if show_network:
            draw_network(screen, model, font, inputs)

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
