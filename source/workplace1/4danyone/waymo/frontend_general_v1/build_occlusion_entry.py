from pathlib import Path
G=Path(__file__).resolve().parent
s=(G.parent/'frontend_quality_20261020/motion_unmasked_entry.py').read_text()
a='''        extractor=VitPoseExtractor();keypoints=extractor.extract(str(video),bbx_xys);del extractor
        torch.save(keypoints,cfg.paths.vitpose)
        extractor=Extractor();features=extractor.extract_video_features(str(video),bbx_xys);del extractor
        torch.save(features,cfg.paths.vit_features);torch.cuda.empty_cache()'''
b='''        branch=ROOT.name
        case=json.loads((ROOT.parent/'case.json').read_text())
        if branch=='kp_only':
            cached=Path(case['original_source'])/'motion_pair/runtime/calibrated/preprocess'
            keypoints=torch.load(cached/'vitpose.pt',map_location='cpu',weights_only=True).clone()
            features=torch.load(cached/'vit_features.pt',map_location='cpu',weights_only=True)
            feature_source=str(cached/'vit_features.pt')
        elif branch=='rgb':
            extractor=VitPoseExtractor();keypoints=extractor.extract(str(video),bbx_xys);del extractor
            extractor=Extractor();features=extractor.extract_video_features(str(video),bbx_xys);del extractor
            feature_source='fresh occluded canonical_clip.mp4; no original feature load'
        else:raise ValueError('Unknown controlled branch')
        raw_keypoints=keypoints.clone();torch.save(raw_keypoints,out/'detector_before_known_occluder.pt')
        occ=np.load(ROOT.parent/'evidence_masks.npz')['occluders']
        xy=np.rint(keypoints[...,:2].cpu().numpy()).astype(int);hh,ww=occ.shape[1:]
        inside=(xy[...,0]>=0)&(xy[...,0]<ww)&(xy[...,1]>=0)&(xy[...,1]<hh)
        hidden=inside&(occ[np.arange(len(occ))[:,None],xy[...,1].clip(0,hh-1),xy[...,0].clip(0,ww-1)]>0)
        keypoints[torch.from_numpy(hidden).to(keypoints.device)]=0
        torch.save(keypoints,cfg.paths.vitpose);torch.save(features,cfg.paths.vit_features);torch.cuda.empty_cache()
        save(out/'feature_provenance.json',{'branch':branch,'feature_source':feature_source,
            'input_video_sha256':sha(video),'keypoints_sha256':sha(Path(cfg.paths.vitpose)),
            'features_sha256':sha(Path(cfg.paths.vit_features)),'suppressed_keypoints':int(hidden.sum()),
            'known_occluder_is_oracle_test_condition':True,
            'original_unoccluded_visual_features_consumed':branch=='kp_only',
            'RGB_branch_recomputes_all_features':branch=='rgb'})'''
assert a in s;s=s.replace(a,b)
s=s.replace("'state':'unmasked_motion_inferred_projection_review_pending'","'state':'synthetic_occlusion_motion_inferred_review_pending'")
s=s.replace("'Unsegmented source with identical dataset-identity boxes; inference diagnostic only'","'Controlled synthetic occlusion; known occluder mask; kp_only retains original image evidence; not a real-world occlusion detector'")
s=s.replace("'changes':['K from cropped estimated focal to per-frame calibrated undistorted crop K',\n            'identity rotation increments to actual exposure-aligned camera rotation increments',\n            'static-camera postprocessing to moving-camera postprocessing'],\n        'same':['input pixels','target boxes','keypoint/image features','checkpoint'],", "'changes':['synthetic RGB occlusion or keypoint-only masking as recorded in feature_provenance.json'],\n        'same':['target boxes','camera calibration','checkpoint','timestamps'],")
(G/'occlusion_motion.py').write_text(s)
