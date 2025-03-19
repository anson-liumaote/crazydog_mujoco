import os
import copy
import torch
import torch.nn.functional as F

class PolicyExporterHIM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.estimator = copy.deepcopy(actor_critic.estimator.encoder)

    def forward(self, obs_history):
        parts = self.estimator(obs_history)[:, 0:19]
        vel, z = parts[..., :3], parts[..., 3:]
        z = F.normalize(z, dim=-1, p=2.0)
        return self.actor(torch.cat((obs_history[:, 0:45], vel, z), dim=1))

    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_1.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)

def export_policy_as_onnx(actor_critic, path):
    """Only for Him model"""
    if hasattr(actor_critic, 'estimator'):
        exporter = PolicyExporterHIM(actor_critic)
        tensor_x = torch.rand((1, 270), dtype=torch.float32) # check with actual layer by https://netron.app/ 
        onnx_program = torch.onnx.dynamo_export(exporter, tensor_x) 
        onnx_program.save("policy.onnx")
    # else: 
    #     os.makedirs(path, exist_ok=True)
    #     path = os.path.join(path, 'policy_1.pt')
    #     model = copy.deepcopy(actor_critic.actor).to('cpu')
    #     traced_script_module = torch.jit.script(model)
    #     traced_script_module.save(path)