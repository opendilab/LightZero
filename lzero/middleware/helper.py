from lzero.policy import visit_count_temperature


def lr_scheduler(cfg, policy):
    max_step = cfg.policy.threshold_training_steps_for_final_lr

    def _schedule(ctx):
        if cfg.policy.lr_piecewise_constant_decay:
            step = ctx.train_iter * cfg.policy.update_per_collect
            if step < 0.5 * max_step:
                policy._optimizer.lr = 0.2
            elif step < 0.75 * max_step:
                policy._optimizer.lr = 0.02
            else:
                policy._optimizer.lr = 0.002

    return _schedule


def temperature_handler(cfg, env):

    def _handle(ctx):
        step = ctx.train_iter * cfg.policy.update_per_collect
        temperature = visit_count_temperature(
            cfg.policy.manual_temperature_decay, 0.25, cfg.policy.threshold_training_steps_for_final_temperature, step
        )
        ctx.collect_kwargs['temperature'] = temperature

    return _handle
