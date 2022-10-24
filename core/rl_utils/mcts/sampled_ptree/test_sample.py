from torch.distributions import Normal, Independent
import torch


# policy_logits = {'mu': torch.randn([1, 2]), 'sigma': abs(torch.randn([1, 2]))}
policy_logits = {'mu': torch.randn([1, 2]), 'sigma': torch.zeros([1, 2])+1e-7}


num_of_sampled_actions = 20

(mu, sigma) = policy_logits['mu'], policy_logits['sigma']
dist = Independent(Normal(mu, sigma), 1)
# dist = Normal(mu, sigma)

print(dist.batch_shape, dist.event_shape)

sampled_actions = dist.sample(torch.tensor([num_of_sampled_actions]))

log_prob = dist.log_prob(sampled_actions)
# log_prob = dist.log_prob(sampled_actions).unsqueeze(-1)