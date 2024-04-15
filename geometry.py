"""Module implementing tools to examine the geometry of a model."""
import sys
import torch
from torch import nn
from torch.autograd.functional import jacobian, hessian
from tqdm import tqdm
# from scipy.integrate import solve_ivp
from torchdiffeq import odeint, odeint_event


class GeometricModel(object):
    
    def __init__(self,
                 network: nn.Module,
                 network_score: nn.Module,
                 verbose: bool=False,
    ) -> None:

        super(GeometricModel, self).__init__()
        self.network = network
        self.network_score = network_score
        # self.network.eval()
        self.verbose = verbose
        self.device = next(self.network.parameters()).device
        self.dtype = next(self.network.parameters()).dtype


    def proba(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:

        if len(eval_point.shape) == 3:  # TODO: trouver un truc plus propre
            eval_point = eval_point.unsqueeze(0)
        p = torch.exp(self.network(eval_point))
        if self.verbose: print(f"proba: {p}")
        return p

    def score(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:
        
        if len(eval_point.shape) == 3:  # TODO: trouver un truc plus propre
            eval_point = eval_point.unsqueeze(0)
        
        return self.network_score(eval_point)


    def grad_proba(
        self,
        eval_point: torch.Tensor,
        wanted_class: int, 
    ) -> torch.Tensor:

        j = jacobian(self.proba, eval_point).squeeze(0)

        grad_proba = j[wanted_class, :]

        return grad_proba


    def jac_proba_true_xor(self, x):
        W_1 = self.network[0].weight
        b_1 = self.network[0].bias
        W_2 = self.network[2].weight
        p = self.proba(x)
        P = torch.diag_embed(p, dim1=1)
        pp = torch.einsum("...i, ...j -> ...ij", p, p)
        T = torch.heaviside(x @ W_1.T + b_1, torch.zeros_like(b_1))
        return torch.einsum(
            "...ik, ...kh, ...h, ...hj -> ...ij",
            P - pp, W_2, T, W_1
        )

    def jac_proba(
        self,
        eval_point: torch.Tensor,
        create_graph: bool=False
    ) -> torch.Tensor:
        """Function computing the matrix ∂_l p_a 

        Args:
            eval_point (torch.Tensor): Batch of points of the input space at
            which the expression is evaluated.
            create_graph (bool, optional): If ``True``, the Jacobian will be
                computed in a differentiable manner.

        Returns:
            torch.Tensor: tensor ∂_l p_a with dimensions (bs, a, l)
        """

        if self.verbose:
            print(f"shape of eval_point = {eval_point.shape}")
            print(f"shape of output = {self.proba(eval_point).shape}")
        j = jacobian(self.proba, eval_point, create_graph=create_graph) # TODO: vérifier dans le cadre non batched
        if self.verbose: print(f"shape of j before reshape = {j.shape}")
        j = j.sum(2)  # 2 is the batch dimension for dx when the output of the net is (bs, c) and because there is no interactions between batches in the derivative we can sum over this dimension to retrieve the only non zero components.
        j = j.flatten(2)
        if self.verbose: print(f"shape of j after reshape = {j.shape}")

        return j
    
    def jac_score(
        self,
        eval_point: torch.Tensor,
        create_graph: bool=False
    ) -> torch.Tensor:
        """Function computing the matrix ∂_l s_a 

        Args:
            eval_point (torch.Tensor): Batch of points of the input space at
            which the expression is evaluated.
            create_graph (bool, optional): If ``True``, the Jacobian will be
                computed in a differentiable manner.

        Returns:
            torch.Tensor: tensor ∂_l s_a with dimensions (bs, a, l)
        """

        if self.verbose:
            print(f"shape of eval_point = {eval_point.shape}")
            print(f"shape of output = {self.proba(eval_point).shape}")
        j = jacobian(self.score, eval_point, create_graph=create_graph)
        if self.verbose: print(f"shape of j before reshape = {j.shape}")
        
        j = j.sum(2)  # 2 is the batch dimension for dx when the output of the net is (bs, c) and because there is no interactions between batches in the derivative we can sum over this dimension to retrieve the only non zero components.
        j = j.flatten(2)
        if self.verbose: print(f"shape of j after reshape = {j.shape}")
        
        return j

        
    def test_jac_proba(
        self,
        eval_point: torch.Tensor,
    ) -> None:
        J_true = self.jac_proba_true_xor(eval_point)
        J = self.jac_proba(eval_point)
        p = self.proba(eval_point)
        P = torch.diag_embed(p, dim1=1)
        pp = torch.einsum("...i, ...j -> ...ij", p, p)
        J_from_score = torch.einsum(
            "...ik, ...kj -> ...ij",
            P - pp, self.jac_score(eval_point)
        )
        good_estimate = torch.isclose(J, J_true).all()
        print(f"Is jac_proba a good estimate for the jacobian?\
                {'Yes' if good_estimate else 'No'}\n \
                Error mean = {(J_true-J).abs().mean()}\n \
                Max error = {(J_true-J).abs().max()} out of {max(J_true.abs().max(), J.abs().max())}")
        
        good_estimate = torch.isclose(J, J_from_score).all()
        print(f"Is jac_from_score a good estimate for the jacobian?\
                {'Yes' if good_estimate else 'No'}\n \
                Error mean = {(J_from_score-J).abs().mean()}\n \
                Max error = {(J_from_score-J).abs().max()} out of {max(J_from_score.abs().max(), J.abs().max())}")

    
    def local_data_matrix(
        self,
        eval_point: torch.Tensor,
        create_graph: bool=False,
        regularisation: bool=False,
    ) -> torch.Tensor:
        """Function computing the Fisher metric wrt the input of the network. 

        Args:
            eval_point (torch.Tensor): Batch of points of the input space at
            which the expression is evaluated.
            create_graph (bool, optional): If ``True``, the Jacobian will be
                computed in a differentiable manner.

        Returns:
            torch.Tensor: tensor g_ij with dimensions (bs, i, j).
        """
        
        J_s = self.jac_score(eval_point, create_graph=create_graph)
        p = self.proba(eval_point)
        P = torch.diag_embed(p, dim1=1)
        pp = torch.einsum("zi,zj -> zij", p, p)
        
        G = torch.einsum("zji, zjk, zkl -> zil", J_s, (P - pp), J_s)
        
        if not regularisation:
            return G

        C = p.shape[-1]
        eigenvalues, eigenvectors = torch.linalg.eigh(G)
        eps = eigenvalues[..., - (C - 1)] / 2

        epsI = torch.einsum("z, ij -> zij", eps, torch.eye(G.shape[-1]))
        epsI[..., - (C - 1):] = 0
        epsKernel = torch.einsum("zij, zjk, zlk -> zil", eigenvectors, epsI, eigenvectors)
        
        return G + epsKernel
    

    def hessian_gradproba(
        self, 
        eval_point: torch.Tensor,
        method: str= 'torch_hessian' # 'relu_optim', 'double_jac', 'torch_hessian'
    ) -> torch.Tensor:
        """Function computing H(p_a)𝛁p_b 

        Args:
            eval_point (torch.Tensor): Batch of points of the input space at
            which the expression is evaluated.
            method (str): Method to compute the hessian:
                - relu_optim: to use only if ReLU network.
                - double_jac: uses double jacobian (slow).
                - torch_hessian: uses torch.autograd.functional.hessian (less slow).

        Returns:
            torch.Tensor: Tensor (H(p_a)𝛁p_b)_l with dimensions (bs, a,b,l).
        """

        if method == 'double_jac':
            J_p = self.jac_proba(eval_point)
            def J(x): return self.jac_proba(x, create_graph=True)
            H_p = jacobian(J, eval_point).sum(3).flatten(3)  # 3 is the batch dimension for dx when the output of the net is (bs, c) and because there is no interactions between batches in the derivative we can sum over this dimension to retrieve the only non zero components.
            h_grad_p = torch.einsum("zalk, zbk -> zabl", H_p, J_p)
            return  h_grad_p

        elif method == 'torch_hessian':
            J_p = self.jac_proba(eval_point)
            shape = self.proba(eval_point).shape
            H_p = []
            for bs, point in enumerate(eval_point):
                H_list = []
                for class_index in tqdm(range(shape[1])):
                    h_p_i = hessian(lambda x: self.proba(x)[0, class_index], point)
                    h_p_i = h_p_i.flatten(len(point.shape))
                    h_p_i = h_p_i.flatten(end_dim=-2)
                    H_list.append(h_p_i)
                H_p.append(torch.stack(H_list))
            H_p = torch.stack(H_p)
            # H_list = torch.stack([torch.stack([hessian(lambda x: self.proba(x)[bs, i], eval_point[bs]) for i in range(shape[1])]) for bs in range(shape[0])])
            h_grad_p = torch.einsum("zalk, zbk -> zabl", H_p, J_p)
            return  h_grad_p
            
        elif method == 'relu_optim':
            J_p = self.jac_proba(eval_point)
            J_s = self.jac_score(eval_point)
            P = self.proba(eval_point)
            C = P.shape[-1]
            I = torch.eye(C).unsqueeze(0)
            N = P.unsqueeze(-2).expand(-1, C, -1)
            
            """Compute """
            first_term = torch.einsum("zbi, zki, zak, zal -> zabl", J_p, J_s, (I-N), J_p) 
            
            """Compute """
            second_term = torch.einsum("za, zbi, zki, zkl -> zabl", P, J_p, J_s, J_p )
            
            return first_term - second_term

    
    
    def lie_bracket(
        self,
        eval_point: torch.Tensor,
        approximation: bool=False,
    ) -> torch.Tensor:
        """Function computing [𝛁p_a, 𝛁p_b] = H(p_b)𝛁p_a - H(p_a)𝛁p_b

        Args:
            eval_point (torch.Tensor): Batch of points of the input space at
            which the expression is evaluated.

        Returns:
            torch.Tensor: Tensor [𝛁p_a, 𝛁p_b]_l with dimensions (bs, a,b,l)
        """

        if approximation:
            J_x = self.jac_proba(eval_point)
            new_point = eval_point.unsqueeze() + J_x
            raise NotImplementedError
        
        H_grad = self.hessian_gradproba(eval_point)
        
        return H_grad.transpose(-2, -3) - H_grad
    

    def jac_dot_product(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:
        """Function computing ∂_i(∇p_a^t ∇p_b).

        Args:
            eval_point (torch.Tensor): Batch of points of the input space at
            which the expression is evaluated.

        Returns:
            torch.Tensor: Tensor ∂_i(∇p_a^t ∇p_b) with dimensions (bs, a, b, i).
        """

        H_grad = self.hessian_gradproba(eval_point)

        return H_grad.transpose(-2, -3) + H_grad
    

    def jac_metric(
        self,
        eval_point: torch.Tensor,
        relu_optim: bool=True,  # Else intractable
    ) -> torch.Tensor:
        """Function computing ∂_k G_{i,j}.

        Args:
            eval_point (torch.Tensor): Batch of points of the input space at
            which the expression is evaluated.
            relu_optim (Boolean): Optimization of the computation if using
            only ReLU in the model.

        Returns:
            torch.Tensor: Tensor ∂_k G_{i,j} with dimensions (bs, i, j, k).
        """

        if relu_optim:
            J_s = self.jac_score(eval_point)
            J_p = self.jac_proba(eval_point)
            p = self.proba(eval_point)
            pdp = torch.einsum("...a, ...bk -> ...kab", p, J_p)  # p_a ∂_k p_b
            return torch.einsum(
                        "...ai, ...kab, ...bj -> ...ijk",
                        J_s, torch.diag_embed(J_p.mT) - pdp - pdp.mT, J_s
                    )
        else:
            # raise NotImplementedError
            # self.verbose=True
            def G(x): return self.local_data_matrix(x, create_graph=True)
            if self.verbose:
                print(f"shape of eval_point = {eval_point.shape}")
                print(f"shape of output = {self.proba(eval_point).shape}")
            jac_metric = jacobian(G, eval_point)
            if self.verbose: print(f"shape of j before reshape = {jac_metric.shape}")
            jac_metric = jac_metric.sum(3).flatten(3)  # Before reshape: (bs, i, j, bs_, k)  ∂_k G_{i,j}
            if self.verbose: print(f"shape of j after reshape = {jac_metric.shape}")
            # self.verbose=False
            return jac_metric


    def christoffel(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:
        """Function computing Γ_{i,j}^k.

        Args:
            eval_point (torch.Tensor): Batch of points of the input space at
            which the expression is evaluated.

        Returns:
            torch.Tensor: Tensor Γ_{i,j}^k with dimensions (bs, i, j, k).
        """
        J_G = self.jac_metric(eval_point)
        G = self.local_data_matrix(eval_point)
        B = J_G.permute(0, 3, 1, 2) + J_G.permute(0, 1, 3, 2) - J_G.permute(0, 3, 2, 1)
        # G_inv = torch.linalg.pinv(G.to(torch.double), hermitian=True).to(self.dtype) # input need to be in double
        # TODO garde fou pour quand G devient nulle, ou que G_inv diverge
        try:
            # G shape: (bs, l, k) | B shape: (bs, i, l, j)
            result_lstsq = torch.linalg.lstsq(G.unsqueeze(-3).expand((*G.shape[:-2], B.shape[-3], *G.shape[-2:])), B, rcond=1e-7)
            result_lstsq = result_lstsq.solution / 2
            result_lstsq = result_lstsq.mT # lstsq gives (bs, i, k, j) and we want (bs, i, j, k)
        except:
            print("Warning: lstsq in project_transverse raised an error.")
            result_lstsq = torch.zeros_like(J_G)
        # B_expanded = B.unsqueeze(-1).expand((*B.shape, G.shape[-2])).transpose(-1, -2)
        # result_lstsq = torch.linalg.lstsq(G[...,None, None, :, :].expand(B_expanded.shape), B_expanded).solution / 2
        # result_pinv = torch.einsum("...kl, ...ilj -> ...ijk", G_inv, B) / 2
        return result_lstsq
    
    def project_kernel(
        self,
        eval_point: torch.Tensor,
        direction: torch.Tensor,
    ) -> torch.Tensor:
        J = self.jac_proba(eval_point)
        J_T = J.mT
        kernel_basis = torch.qr(J_T, some=False).Q[:, J_T.shape[1] - 1:]  # we extract the last component since the sum of the column of J_T is equal to zero -> this is only the basis of the kernel of J_T 
        coefficients = torch.linalg.lstsq(kernel_basis, direction).solution
        displacement = torch.mv(kernel_basis, coefficients)
        return displacement
        
    def project_transverse(
        self,
        eval_point: torch.Tensor,
        direction: torch.Tensor,
    ) -> torch.Tensor:
        J = self.jac_proba(eval_point)
        J_T = J.mT
        try:
            coefficients = torch.linalg.lstsq(J_T, direction).solution
            displacement = torch.einsum("zla, za -> zl", J_T, coefficients)
        except:
            print("Warning: lstsq in project_transverse raised an error.")
            displacement = direction
        return displacement

    def geodesic(
        self,
        eval_point: torch.Tensor,
        init_velocity: torch.Tensor,
        euclidean_budget: float=None,
        full_path: bool=False,
        project_leaf: bool=True,
    ) -> torch.Tensor:
        """Compute the geodesic for the FIM's LC connection with initial velocities [init_velocity] at points [eval_point].

        Args:
            eval_point (torch.Tensor): Batch of points of the input space at
            which the expression is evaluated.
            init_velocity (torch.Tensor): Batch of initial velocities for the geodesic.
            euclidean_budget (float, optional): Euclidean budget for the point. Defaults to None.
            full_path (bool, optional): When True, returns the full geodesic path. Defaults to False.
            project_leaf (bool, optional): When True, projects the velocity to the transverse leaf at each step. Defaults to True.

        Returns:
            torch.Tensor: Arrival point of the geodesic with dimensions (bs, i)
        """
        print(f"ɛ={euclidean_budget}")
        if len(init_velocity.shape) > 2:
            init_velocity = init_velocity.flatten(1)
        
        def ode(t, y):
            x, v = y
            christoffel = self.christoffel(x)
            a = -torch.einsum("...i, ...j, ...ijk -> ...k", v, v, christoffel)
            # print(f"|v|={v.norm()}", end='\r')
            if project_leaf:
                v = self.project_transverse(x, v)
                # a = self.project_transverse(x, a)
            # sys.stdout.write("\033[K") 
            # if self.verbose:
            self.iteration += 1
            # print("\033[K", end='\r')
            # print(f"iteration n°{self.iteration}: |v|={v.norm():4e}, |a|={a.norm():4e}", end='\r')
            return (v.reshape(x.shape), a)
        
        if euclidean_budget is None:
            raise NotImplementedError
            y0 = (eval_point, init_velocity) # TODO: wrong dim after bash -> should be flatten ?

            solution_ode = odeint(ode, y0, t=torch.linspace(0., 4., 1000), method="rk4")
            solution_ode_x, solution_ode_v = solution_ode

            return solution_ode_x[-1]

        elif euclidean_budget <= 0.:
            return eval_point

        else:
            self.iteration = 0 
            if self.verbose:
                print(f"eval_point: {eval_point.shape}")
                print(f"init_velocity: {init_velocity.shape}")

            if not full_path:
                print("Geodesic computation starting...")
                solution_ode_x, solution_ode_v = [], []
                for point, vel in tqdm(zip(eval_point, init_velocity)):
                    self.iteration = 0
                    y0 = (point.unsqueeze(0), vel.unsqueeze(0))
                    def euclidean_stop(t, y):
                        x, v = y
                        # print("\033[K", end='\r')
                        # print(f"Iteration n°{self.iteration} - Euclidean norm: {float(euclidean_budget - torch.norm(x - y0[0])):3e}", end='\r')
                        return nn.functional.relu(euclidean_budget - torch.norm(x - y0[0])) * (v.norm() > 1e-7).float()
                    with torch.no_grad():
                        event_t, solution_ode = odeint_event(ode, y0, t0=torch.tensor(0.), event_fn=euclidean_stop, method="euler", options={"step_size": euclidean_budget / 10})
                        # event_t, solution_ode = odeint_event(ode, y0, t0=torch.tensor(0.), event_fn=euclidean_stop, method="adaptive_heun") # too long
                    solution_ode_x.append(solution_ode[0])
                    solution_ode_v.append(solution_ode[1])
                
                solution_ode_x = torch.cat(solution_ode_x, dim=1)
                solution_ode_v = torch.cat(solution_ode_v, dim=1)
                
                return solution_ode_x[-1]
                
            # solution_ivp = solve_ivp(ode, t_span = (0, 2), y0=(eval_point.detach().numpy(), init_velocity.detach().numpy()), method='RK23', events=euclidean_stop if euclidean_budget is not None else None)
            
            # if self.verbose: print(f"event_t: {event_t}")

            solution_ode = odeint(ode, y0, t=torch.linspace(0., int(euclidean_budget * 10), 1000), method="rk4", options={"step_size": euclidean_budget / 100})
            
            # self.verbose = True
            solution_ode_x, solution_ode_v = solution_ode
            if full_path:
                return solution_ode_x.transpose(0, 1)

            # return solution_ode_x[-1]
            
            if self.verbose:
                print(f"solution_ode_x: {solution_ode_x.shape}")
                print(f"solution_ode_v: {solution_ode_v.shape}")
                print(f"0 is initial value ? {torch.allclose(solution_ode_x[0], eval_point)} dist: {torch.dist(solution_ode_x[0], eval_point)}")

            # Get last point exceeding the euclidean budget
            admissible_indices = ((solution_ode_x - eval_point.unsqueeze(0)).flatten(2).norm(dim=-1) <= euclidean_budget)
            last_admissible_index = admissible_indices.shape[0] - 1 - admissible_indices.flip(dims=[0]).int().argmax(dim=0)
            last_admissible_solution_x = torch.diagonal(solution_ode_x[last_admissible_index], dim1=0, dim2=1).movedim(-1, 0)
            print(f"Warning: geodesics stoped before reaching ɛ: {(last_admissible_index == admissible_indices.shape[0] -1).float().mean() * 100:.2f}%")
                
            if self.verbose:
                last_admissible_solution_x_loop = torch.zeros_like(eval_point)
                last_admissible_index_loop = torch.zeros(eval_point.shape[0])

                for i, step in enumerate(solution_ode_x):
                    for j, batch in enumerate(step):
                        if (batch - eval_point[j]).norm() <= euclidean_budget:
                            last_admissible_index_loop[j] = i
                            last_admissible_solution_x_loop[j] = batch
                print(f"2 solutions are the same ? {torch.allclose(last_admissible_solution_x, last_admissible_solution_x_loop)}")
                print(f"2 indices of solutions are the same ? {torch.allclose(last_admissible_index.int(), last_admissible_index_loop.int())}")
                        
            return last_admissible_solution_x
