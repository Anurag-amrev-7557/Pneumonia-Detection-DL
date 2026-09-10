"""
Main entry point for the pneumonia detection application.

Provides a unified command-line interface for model training, fine-tuning,
evaluation, single/batch inference with Grad-CAM, web server, and dashboard.
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

# Setup path
sys.path.insert(0, str(Path(__file__).parent))

from src.config.settings import settings
from src.models.grad_cam import GradCAMVisualizer
from src.models.inference import PneumoniaDetector

# Configure logging
logging.basicConfig(
    level=settings.log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Pneumonia Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a model
  python main.py train --model resnet50 --epochs 50 --batch-size 32

  # Fine-tune the trained model
  python main.py finetune --epochs 15

  # Run inference on a single image
  python main.py predict --image data/test/NORMAL/NORMAL-1003233-0001.jpeg --explain

  # Run batch inference on a folder
  python main.py predict-batch --directory data/test/PNEUMONIA

  # Evaluate model on test set
  python main.py evaluate --test-dir data/test

  # Launch FastAPI PACS Diagnostic Workstation
  python main.py server --port 8000

  # Launch Streamlit dashboard
  python main.py dashboard --port 8501
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train deep learning model')
    train_parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('data'),
        help='Dataset directory (default: data)'
    )
    train_parser.add_argument(
        '--model',
        default='resnet50',
        choices=['resnet50', 'vgg16', 'custom_cnn', 'inceptionv3'],
        help='Model architecture (default: resnet50)'
    )
    train_parser.add_argument('--epochs', type=int, default=50, help='Max epochs (default: 50)')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Batch size (default: 32)')
    train_parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate (default: 0.001)')
    train_parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('models/current'),
        help='Output directory (default: models/current)'
    )

    # Finetune command
    ft_parser = subparsers.add_parser('finetune', help='Fine-tune trained ResNet50 model')
    ft_parser.add_argument(
        '--model-path',
        default='models/current/best_model.h5',
        help='Path to base model checkpoint'
    )
    ft_parser.add_argument('--epochs', type=int, default=15, help='Fine-tuning epochs (default: 15)')
    ft_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    ft_parser.add_argument('--learning-rate', type=float, default=1e-5, help='Fine-tuning learning rate (default: 1e-5)')
    ft_parser.add_argument('--output-dir', default='models/current', help='Output directory')

    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model on test dataset')
    eval_parser.add_argument(
        '--model',
        default=str(settings.inference.model_path),
        help='Path to model file'
    )
    eval_parser.add_argument(
        '--test-dir',
        type=Path,
        default=Path('data/test'),
        help='Path to test directory containing NORMAL and PNEUMONIA folders'
    )

    # Server command
    server_parser = subparsers.add_parser('server', help='Start FastAPI PACS diagnostic workstation server')
    server_parser.add_argument('--host', default=settings.ui.host, help='Server host (default: 0.0.0.0)')
    server_parser.add_argument('--port', type=int, default=settings.ui.port, help='Server port (default: 8000)')
    server_parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    # Dashboard command
    dash_parser = subparsers.add_parser('dashboard', help='Launch Streamlit dashboard')
    dash_parser.add_argument('--port', type=int, default=8501, help='Port (default: 8501)')

    # Single prediction command
    predict_parser = subparsers.add_parser('predict', help='Run inference on single image')
    predict_parser.add_argument('--image', required=True, help='Path to image file')
    predict_parser.add_argument(
        '--model',
        default=str(settings.inference.model_path),
        help='Path to model file'
    )
    predict_parser.add_argument('--explain', action='store_true', help='Generate Grad-CAM explanation')
    predict_parser.add_argument('--output-dir', help='Directory to save Grad-CAM explanation image')

    # Batch prediction command
    batch_parser = subparsers.add_parser('predict-batch', help='Run inference on directory')
    batch_parser.add_argument('--directory', required=True, help='Directory containing images')
    batch_parser.add_argument(
        '--model',
        default=str(settings.inference.model_path),
        help='Path to model file'
    )
    batch_parser.add_argument('--output', help='Output JSON file for results')

    # Version command
    subparsers.add_parser('version', help='Show version')

    args = parser.parse_args()

    try:
        if args.command == 'train':
            _run_train(args)
        elif args.command == 'finetune':
            _run_finetune(args)
        elif args.command == 'evaluate':
            _run_evaluate(args)
        elif args.command == 'server':
            _run_server(args)
        elif args.command == 'dashboard':
            _run_dashboard(args)
        elif args.command == 'predict':
            _run_predict(args)
        elif args.command == 'predict-batch':
            _run_batch_predict(args)
        elif args.command == 'version':
            _show_version()
        else:
            parser.print_help()

    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)


def _run_train(args):
    """Run model training."""
    from src.models.training_pipeline import train
    train(
        data_dir=args.data_dir,
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
    )


def _run_finetune(args):
    """Run model fine-tuning."""
    from src.models.fine_tune_pipeline import fine_tune
    fine_tune(
        model_path=args.model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
    )


def _run_evaluate(args):
    """Run model evaluation on test set."""
    from scripts.evaluate_model import evaluate
    evaluate(
        test_dir=args.test_dir,
        model_path=args.model,
    )


def _run_server(args):
    """Run FastAPI PACS workstation server."""
    import uvicorn
    logger.info(f"Starting PULMO·AI PACS Diagnostic Server on http://{args.host}:{args.port} ...")
    uvicorn.run(
        "src.api.server:app",
        host=args.host,
        port=args.port,
        reload=args.debug,
    )


def _run_dashboard(args):
    """Launch Streamlit dashboard."""
    cmd = ["streamlit", "run", "src/ui/streamlit_app.py", "--server.port", str(args.port)]
    subprocess.run(cmd)


def _run_predict(args):
    """Run single image prediction."""
    logger.info(f"Running prediction on {args.image}")
    detector = PneumoniaDetector(model_path=args.model)
    result = detector.predict_image(args.image)

    # Print results
    print("\n" + "=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)
    print(f"Image:          {args.image}")
    print(f"Prediction:     {result['class']}")
    print(f"Confidence:     {result['confidence']:.2%}")
    print(f"Is Pneumonia:   {result['is_pneumonia']}")
    print(f"Is Confident:   {result['is_confident']}")

    if result.get('probabilities'):
        print("\nClass Probabilities:")
        for cls, prob in result['probabilities'].items():
            print(f"  {cls}: {prob:.4f}")

    if args.explain:
        output_dir = Path(args.output_dir) if args.output_dir else Path("scratch")
        output_dir.mkdir(parents=True, exist_ok=True)
        img_p = Path(args.image)
        output_path = output_dir / f"{img_p.stem}_gradcam.png"
        try:
            grad_cam = GradCAMVisualizer(detector.model)
            grad_cam.visualize_and_save(str(img_p), str(output_path))
            print(f"\nGrad-CAM Saved: {output_path}")
        except Exception as e:
            logger.warning(f"Grad-CAM generation failed: {e}")

    print("=" * 60 + "\n")


def _run_batch_predict(args):
    """Run batch prediction."""
    logger.info(f"Running batch prediction on directory {args.directory}")
    detector = PneumoniaDetector(model_path=args.model)
    batch_dir = Path(args.directory)

    image_extensions = ('.jpg', '.jpeg', '.png', '.gif')
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(batch_dir.glob(f'*{ext}'))
        image_paths.extend(batch_dir.glob(f'*{ext.upper()}'))

    if not image_paths:
        logger.warning(f"No images found in {args.directory}")
        return

    logger.info(f"Found {len(image_paths)} images")
    results = []
    for image_path in image_paths:
        try:
            result = detector.predict_image(str(image_path))
            result['image'] = str(image_path.relative_to(batch_dir))
            results.append(result)
        except (ValueError, RuntimeError, OSError) as e:
            logger.error(f"Failed to process {image_path}: {e}")

    pneumonia_count = sum(1 for r in results if r['is_pneumonia'])
    normal_count = len(results) - pneumonia_count

    print("\n" + "=" * 60)
    print("BATCH PREDICTION SUMMARY")
    print("=" * 60)
    print(f"Total Images:       {len(results)}")
    print(f"Pneumonia Cases:    {pneumonia_count} ({pneumonia_count / len(results) * 100:.1f}%)")
    print(f"Normal Cases:       {normal_count} ({normal_count / len(results) * 100:.1f}%)")
    print(f"Average Confidence: {np.mean([r['confidence'] for r in results]):.2%}")
    print("=" * 60 + "\n")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results to {output_path}")


def _show_version():
    """Show version information."""
    import src
    print(f"Pneumonia Detection System v{src.__version__}")


if __name__ == '__main__':
    main()
