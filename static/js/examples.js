// Function to load example content from files
function loadExampleContent() {
  const exampleContainers = document.querySelectorAll('.example-container');
  
  exampleContainers.forEach(container => {
    const baselinePath = container.getAttribute('data-baseline');
    const frePath = container.getAttribute('data-fre');
    const baselineContainer = container.querySelector('.baseline-content');
    const freContainer = container.querySelector('.fre-content');
    
    if (baselinePath && baselineContainer) {
      fetch(baselinePath)
        .then(response => response.text())
        .then(text => {
          baselineContainer.textContent = text;
        })
        .catch(error => {
          console.error('Error loading baseline example:', error);
        });
    }
    
    if (frePath && freContainer) {
      fetch(frePath)
        .then(response => response.text())
        .then(text => {
          freContainer.textContent = text;
        })
        .catch(error => {
          console.error('Error loading FRE example:', error);
        });
    }
  });
}

// Load examples when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', loadExampleContent); 