from src.ensemble import SlopDetectorEnsemble

# Test with dummy examples
detector = SlopDetectorEnsemble()

test_cases = [
    "This product is amazing! Perfect for anyone looking for quality and value. Highly recommended!",
    "honestly kinda mid ngl. battery died after 6hrs which is trash. my old phone lasted way longer lol",
    "In today's digital landscape, it's worth noting that this innovative solution leverages cutting-edge technology.",
]

for i, text in enumerate(test_cases, 1):
    print(f"\n{'='*60}")
    print(f"Test {i}: {text[:60]}...")
    print(f"{'='*60}")
    
    result = detector.detect(text)
    
    print(f"\nVerdict: {result['verdict']}")
    print(f"Score: {result['final_score']}/100")
    print(f"Confidence: {result['confidence']}")
    print(f"\nAgent Scores:")
    for agent, score in result['agent_scores'].items():
        print(f"  {agent:12s}: {score:5.1f}%")