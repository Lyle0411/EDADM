#开发时间: 2023/1/8 10:56

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data


class Diffusion_Design:
    def __init__(self, performance, N, p, num_epoch=4000):
        Nor_temp = np.max(performance, axis=0)
        performance = performance / Nor_temp
        Configuration_size = performance.shape[1]

        print("shape of configuration:", np.shape(performance))
        if performance.shape[0] % 2:
            performance = np.append(performance, [performance[-1]], axis = 0)

        dataset = torch.Tensor(performance).float()

        num_steps = 100

        #制定每一步的beta
        betas = torch.linspace(-6,6, num_steps)
        betas = torch.sigmoid(betas)*(0.5e-2 - 1e-5)+1e-5

        #计算alpha、alpha_prod、alpha_prod_previous、alpha_bar_sqrt等变量的值
        alphas = 1-betas
        alphas_prod = torch.cumprod(alphas,0)
        alphas_prod_p = torch.cat([torch.tensor([1]).float(),alphas_prod[:-1]],0)
        alphas_bar_sqrt = torch.sqrt(alphas_prod)
        one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
        one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

        #断言参数同型
        assert alphas.shape==alphas_prod.shape==alphas_prod_p.shape==\
        alphas_bar_sqrt.shape==one_minus_alphas_bar_log.shape\
        ==one_minus_alphas_bar_sqrt.shape

        def q_x(x_0,t):
            """可以基于x[0]得到任意时刻t的x[t]"""
            noise = torch.randn_like(x_0)
            alphas_t = alphas_bar_sqrt[t]
            alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
            return (alphas_t * x_0 + alphas_1_m_t * noise)#在x_0的基础上添加噪声



        class MLPDiffusion(nn.Module):
            def __init__(self, n_steps, num_units=128):
                super(MLPDiffusion, self).__init__()

                self.linears = nn.ModuleList(
                    [
                        nn.Linear(Configuration_size, num_units),
                        nn.ReLU(),
                        nn.Linear(num_units, num_units),
                        nn.ReLU(),
                        nn.Linear(num_units, num_units),
                        nn.ReLU(),
                        nn.Linear(num_units, Configuration_size),
                    ]
                )
                self.step_embeddings = nn.ModuleList(
                    [
                        nn.Embedding(n_steps, num_units),
                        nn.Embedding(n_steps, num_units),
                        nn.Embedding(n_steps, num_units),
                    ]
                )

            def forward(self, x, t):
                #         x = x_0
                for idx, embedding_layer in enumerate(self.step_embeddings):
                    t_embedding = embedding_layer(t)
                    x = self.linears[2 * idx](x)
                    x += t_embedding
                    x = self.linears[2 * idx + 1](x)

                x = self.linears[-1](x)

                return x

        def diffusion_loss_fn(model, x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps):
            """对任意时刻t进行采样计算loss"""
            batch_size = x_0.shape[0]

            # 对一个batchsize样本生成随机的时刻t
            t = torch.randint(0, n_steps, size=(batch_size // 2,))
            t = torch.cat([t, n_steps - 1 - t], dim=0)

            t = t.unsqueeze(-1)

            # x0的系数

            a = alphas_bar_sqrt[t]

            # eps的系数
            aml = one_minus_alphas_bar_sqrt[t]

            # 生成随机噪音eps
            e = torch.randn_like(x_0)

            # 构造模型的输入

            x = x_0 * a + e * aml


            # 送入模型，得到t时刻的随机噪声预测值
            output = model(x, t.squeeze(-1))

            # 与真实噪声一起计算误差，求平均值

            return (e - output).square().mean()

        def p_sample_loop(model, shape, n_steps, betas, one_minus_alphas_bar_sqrt):
            """从x[T]恢复x[T-1]、x[T-2]|...x[0]"""
            cur_x = torch.randn(shape)
            x_seq = [cur_x]
            for i in reversed(range(n_steps)):
                cur_x = p_sample(model, cur_x, i, betas, one_minus_alphas_bar_sqrt)
                x_seq.append(cur_x)
            return x_seq


        def p_sample(model, x, t, betas, one_minus_alphas_bar_sqrt):
            """从x[T]采样t时刻的重构值"""
            t = torch.tensor([t])

            coeff = betas[t] / one_minus_alphas_bar_sqrt[t]

            eps_theta = model(x, t)

            mean = (1 / (1 - betas[t]).sqrt()) * (x - (coeff * eps_theta))

            z = torch.randn_like(x)
            sigma_t = betas[t].sqrt()

            sample = mean + sigma_t * z

            return (sample)

        seed = 1234


        class EMA():
            """构建一个参数平滑器"""

            def __init__(self, mu=0.01):
                self.mu = mu
                self.shadow = {}

            def register(self, name, val):
                self.shadow[name] = val.clone()

            def __call__(self, name, x):
                assert name in self.shadow
                new_average = self.mu * x + (1.0 - self.mu) * self.shadow[name]
                self.shadow[name] = new_average.clone()
                return new_average

        print('Training model...')
        batch_size = 128
        dataloader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        plt.rc('text', color='blue')

        model = MLPDiffusion(num_steps)  # 输出维度是2，输入是x和step
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for t in range(num_epoch):
            for idx, batch_x in enumerate(dataloader):
                loss = diffusion_loss_fn(model, batch_x, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, num_steps)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()

            if (t % 100 == 0):
                print(loss)
                x_seq = p_sample_loop(model, dataset.shape, num_steps, betas, one_minus_alphas_bar_sqrt)


                # fig, axs = plt.subplots(1, 10, figsize=(28, 3))
                # for i in range(1, 11):
                #     cur_x = x_seq[i * 10].detach()
                #     axs[i - 1].scatter(cur_x[:, 0], cur_x[:, 1], color='red', edgecolor='white')
                #     axs[i - 1].set_axis_off()
                #     axs[i - 1].set_title('$q(\mathbf{x}_{' + str(i * 10) + '})$')


        x_seq = p_sample_loop(model, [p*N, performance.shape[1]], num_steps, betas, one_minus_alphas_bar_sqrt)
        self.x_0_new = np.array(x_seq[-1].detach()) * Nor_temp


