# from modules.trainer.reg_trainer import RegTrainer
# from modules.trainer.joint_registration_regression_trainer import JointRegistrationRegressionTrainer
# from modules.trainer.LMA_trainer import LMATrainer
# from modules.trainer.strainmat_pred_trainer import StrainmatPredTrainer
# from modules.trainer.strainmat_LMA_trainer import StrainmatLMATrainer
# from modules.trainer.joint_registration_strainmat_LMA import JointRegisterStrainmatLMATrainer
from modules.trainer.base_trainer import BaseTrainer
from modules.trainer.regression_trainer import RegressionTrainer
from modules.trainer.registration_trainer import RegTrainer
# from modules.trainer.ae_trainer import AETrainer
# from modules.trainer.DENSE_disp_pred_trainer import DENSEDispPredTrainer
# from modules.trainer.DENSE_disp_pred_naive_trainer import DENSEDispPredNaiveTrainer
# from modules.trainer.DENSE_disp_cond_gene_trainer import DENSEDispCondGeneTrainer
# from modules.trainer.DENSE_disp_cond_pred_trainer import DENSEDispCondPredTrainer
# from modules.trainer.DENSE_disp_regression_trainer import DENSEDispRegressionTrainer
# from modules.trainer.DENSE_latent_denoiser_trainer import DENSELatentDenoiserTrainer
# from modules.trainer.DENSE_bio_info_motion_tracking_trainer import DENSEBioInfoMotionTrackingTrainer
# from modules.trainer.DENSE_bio_info_motion_tracking_VAE_trainer import DENSEBioInfoMotionTrackingVAETrainer

# from modules.trainer.DENSE_disp_pred_alter_trainer import DENSEDispPredAlterTrainer

def build_trainer(trainer_config, device=None, full_config=None):
    trainer_scheme = trainer_config['scheme']
    if trainer_scheme.lower() in ['regression']:
        return RegressionTrainer(trainer_config, device, full_config)
    if trainer_scheme.lower() in ['reg', 'registration']:
        return RegTrainer(trainer_config, device, full_config)
    # elif trainer_scheme.lower() in ['ae',' autoencoder']:
    #     return AETrainer(trainer_config, device, full_config)
    # elif trainer_scheme == 'DENSE_disp_pred':
    #     return DENSEDispPredTrainer(trainer_config, device, full_config)
    # elif trainer_scheme == 'DENSE_disp_pred_naive':
    #     return DENSEDispPredNaiveTrainer(trainer_config, device, full_config)
    # elif trainer_scheme == 'DENSE_disp_cond_pred':
    #     return DENSEDispCondPredTrainer(trainer_config, device, full_config)
    # elif trainer_scheme == 'DENSE_disp_cond_gene':
    #     return DENSEDispCondGeneTrainer(trainer_config, device, full_config)
    # elif trainer_scheme == 'DENSE_disp_regression':
    #     return DENSEDispRegressionTrainer(trainer_config, device, full_config)
    # elif trainer_scheme == 'DENSE_latent_denoiser':
    #     return DENSELatentDenoiserTrainer(trainer_config, device, full_config)
    # elif trainer_scheme == 'DENSE_bio_info_motion_tracking':
    #     return DENSEBioInfoMotionTrackingTrainer(trainer_config, device, full_config)
    # elif trainer_scheme == 'DENSE_bio_info_motion_tracking_VAE':
    #     return DENSEBioInfoMotionTrackingVAETrainer(trainer_config, device, full_config)
    # elif trainer_scheme == 'DENSE_disp_pred_alter':
    #     return DENSEDispPredAlterTrainer(trainer_config, device, full_config)
    # elif trainer_scheme == 'LMA':
    #     return LMATrainer(trainer_config, device, full_config)
    # elif trainer_scheme == 'joint_registration_regression':
    #     return JointRegistrationRegressionTrainer(trainer_config, device, full_config)
    # elif trainer_scheme == 'strainmat_pred':
    #     return StrainmatPredTrainer(trainer_config, device, full_config)
    # elif trainer_scheme == 'strainmat_LMA':
    #     return StrainmatLMATrainer(trainer_config, device, full_config)
    # elif trainer_scheme == 'joint_registration_strainmat_LMA':
    #     return JointRegisterStrainmatLMATrainer(trainer_config, device, full_config)
    else:
        raise NotImplementedError(f"trainer scheme {trainer_scheme} not implemented")