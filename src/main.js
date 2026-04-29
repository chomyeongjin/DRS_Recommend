import './style.css';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader';
import gsap from 'gsap';

// Mock Data
const stocks = [
  { id: 'TSLA', name: 'Tesla Inc', symbol: 'TSLA', price: '$210.20', change: '+5.10%', color: '#E82127', rank: 2, height: 1.5, pos: -2.5, podiumColor: 0xC0C0C0 }, // 2nd
  { id: 'AAPL', name: 'Apple Inc', symbol: 'AAPL', price: '$180.50', change: '+2.40%', color: '#555555', rank: 1, height: 2.2, pos: 0, podiumColor: 0xFFD700 }, // 1st
  { id: 'GOOG', name: 'Alphabet', symbol: 'GOOG', price: '$135.40', change: '+1.20%', color: '#4285F4', rank: 3, height: 1.0, pos: 2.5, podiumColor: 0xcd7f32 }, // 3rd
];

// Scene Setup
const canvas = document.querySelector('#app-canvas');
const scene = new THREE.Scene();
scene.background = new THREE.Color('#090a0f'); // Explicitly set space background
scene.fog = new THREE.FogExp2(0x090a0f, 0.025); // Dark space fog

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(0, 3, 12);

const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

// Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.enablePan = false;
controls.minDistance = 5;
controls.maxDistance = 20;
controls.maxPolarAngle = Math.PI / 2 + 0.1; // Don't go too far below ground

controls.target.set(0, 2, 0);
controls.update();

// Lights - Dramatic Space Lighting
const ambientLight = new THREE.AmbientLight(0xffffff, 0.3);
scene.add(ambientLight);

const dirLight = new THREE.DirectionalLight(0xaabbff, 1.2);
dirLight.position.set(5, 10, 5);
dirLight.castShadow = true;
dirLight.shadow.mapSize.width = 1024;
dirLight.shadow.mapSize.height = 1024;
scene.add(dirLight);

const spotLight = new THREE.SpotLight(0xffaadd, 2.0);
spotLight.position.set(-5, 15, -5);
spotLight.angle = Math.PI / 4;
spotLight.penumbra = 0.5;
spotLight.castShadow = true;
scene.add(spotLight);

// Environment - Starry Background
const particlesCount = 2000;
const posArray = new Float32Array(particlesCount * 3);
for (let i = 0; i < particlesCount * 3; i++) {
  posArray[i] = (Math.random() - 0.5) * 80;
}
const particlesGeo = new THREE.BufferGeometry();
particlesGeo.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
const particlesMat = new THREE.PointsMaterial({
  size: 0.1,
  color: 0xffffff,
  transparent: true,
  opacity: 0.8,
  blending: THREE.AdditiveBlending
});
const particlesMesh = new THREE.Points(particlesGeo, particlesMat);
scene.add(particlesMesh);

// Ground - Dark Space Base
const groundMat = new THREE.MeshStandardMaterial({
  color: 0x111122,
  roughness: 0.1,
  metalness: 0.8,
  transparent: true,
  opacity: 0.7
});
const ground = new THREE.Mesh(new THREE.PlaneGeometry(50, 50), groundMat);
ground.rotation.x = -Math.PI / 2;
ground.position.y = -0.5; // Slightly below podium bottoms
ground.receiveShadow = true;
scene.add(ground);

// Helper function to create face logo texture
function createLogoTexture(symbol, bgColor) {
  const c = document.createElement('canvas');
  c.width = 256;
  c.height = 256;
  const ctx = c.getContext('2d');

  // Background
  ctx.fillStyle = bgColor;
  ctx.fillRect(0, 0, 256, 256);

  // Text
  ctx.fillStyle = '#ffffff';
  ctx.font = 'bold 80px sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(symbol, 128, 128);

  return new THREE.CanvasTexture(c);
}

// Top 10 Mock Data for Intro
const top10Stocks = [
  { rank: 1, name: 'Apple Inc', symbol: 'AAPL', price: '$180.50', change: '+2.40%' },
  { rank: 2, name: 'Tesla Inc', symbol: 'TSLA', price: '$210.20', change: '+5.10%' },
  { rank: 3, name: 'Alphabet', symbol: 'GOOG', price: '$135.40', change: '+1.20%' },
  { rank: 4, name: 'NVIDIA Corp', symbol: 'NVDA', price: '$450.10', change: '+3.50%' },
  { rank: 5, name: 'Microsoft', symbol: 'MSFT', price: '$320.00', change: '+1.00%' },
  { rank: 6, name: 'Amazon', symbol: 'AMZN', price: '$140.20', change: '+0.80%' },
  { rank: 7, name: 'Meta Platforms', symbol: 'META', price: '$300.50', change: '+2.10%' },
  { rank: 8, name: 'Berkshire Hathaway', symbol: 'BRK.B', price: '$360.00', change: '+0.50%' },
  { rank: 9, name: 'Visa Inc', symbol: 'V', price: "$240.10", change: '+0.30%' },
  { rank: 10, name: 'Johnson & Johnson', symbol: 'JNJ', price: '$160.20', change: '-0.20%' }
];

// Setup Star Wars Intro
const introContainer = document.getElementById('intro-container');
const introCrawl = document.getElementById('intro-crawl');
const skipBtn = document.getElementById('skip-button');
const uiContainer = document.getElementById('ui-container');
const sidebarContainer = document.getElementById('sidebar-container');
const sidebarList = document.getElementById('sidebar-list');

let crawlHtml = `<div class="intro-title">
  <p>Today's Market</p>
  <h1>TOP 10 RECOMMENDS</h1>
</div><br><br>`;

let sidebarHtml = '';

// 10위부터 1위까지 역순으로 크레딧 생성 (인트로용)
for (let i = 9; i >= 0; i--) {
  const s = top10Stocks[i];
  crawlHtml += `<div class="crawl-item">
    <h2>${s.rank}위. ${s.name} (${s.symbol})</h2>
    <p>가격: ${s.price} | 변동: ${s.change}</p>
  </div><br>`;
}
introCrawl.innerHTML = crawlHtml;

// 1위부터 10위까지 순서대로 사이드바 생성 (메인 화면용)
for (let i = 0; i < 10; i++) {
  const s = top10Stocks[i];
  const changeClass = s.change.startsWith('-') ? 'negative' : 'positive';
  sidebarHtml += `
    <div class="sidebar-item">
      <div class="sidebar-item-left">
        <span class="sidebar-item-rank">#${s.rank}</span>
        <span class="sidebar-item-name">${s.name}</span>
      </div>
      <div class="sidebar-item-right">
        <span class="sidebar-item-price">${s.price}</span>
        <span class="sidebar-item-change ${changeClass}">${s.change}</span>
      </div>
    </div>
  `;
}
sidebarList.innerHTML = sidebarHtml;

let introFinished = false;

function finishIntro() {
  if (introFinished) return;
  introFinished = true;

  // Fade out intro container
  gsap.to(introContainer, {
    opacity: 0, duration: 1, onComplete: () => {
      introContainer.style.display = 'none';
    }
  });

  // Show Main UI & Sidebar
  uiContainer.style.display = 'block';
  sidebarContainer.style.display = 'flex';

  gsap.fromTo(uiContainer, { opacity: 0, y: -20 }, { opacity: 1, y: 0, duration: 1 });
  gsap.fromTo(sidebarContainer, { opacity: 0, x: 20 }, { opacity: 1, x: 0, duration: 1, delay: 0.2 });

  // Build 3D Scene after intro finishes
  buildMainScene();
}

skipBtn.addEventListener('click', finishIntro);
introCrawl.addEventListener('animationend', finishIntro);

// Build Podiums and Characters
const characterMeshes = []; // For Raycasting
let isSceneBuilt = false;

function buildMainScene() {
  if (isSceneBuilt) return;
  isSceneBuilt = true;

  // Build Podiums
  stocks.forEach(stock => {
    const podGeo = new THREE.CylinderGeometry(1, 1.2, stock.height, 32);
    const podMat = new THREE.MeshStandardMaterial({
      color: stock.podiumColor,
      roughness: 0.3,
      metalness: 0.8
    });
    const podium = new THREE.Mesh(podGeo, podMat);
    podium.position.set(stock.pos, stock.height / 2 - 0.5, 0); // Ground is at -0.5
    podium.castShadow = true;
    podium.receiveShadow = true;
    scene.add(podium);
  });

  const gltfLoader = new GLTFLoader();

  // Load the model (cute_alien 사용, stick_man으로 변경 가능)
  gltfLoader.load('/models/cute_alien/scene.gltf', (gltf) => {
    const model = gltf.scene;

    // 모든 메쉬에 그림자 설정
    model.traverse((child) => {
      if (child.isMesh) {
        child.castShadow = true;
        child.receiveShadow = true;
      }
    });

    // 모델의 크기와 중심점을 계산해서 자동으로 1.5 크기로 맞춤
    const box = new THREE.Box3().setFromObject(model);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());

    const targetHeight = 1.5; // 원하는 캐릭터 키
    const scale = targetHeight / size.y;
    model.scale.setScalar(scale);

    // 모델이 단상 중앙에 위치하도록 정렬 (바닥을 0으로)
    model.position.set(
      -center.x * scale,
      -box.min.y * scale,
      -center.z * scale
    );

    // 정렬된 모델을 감싸는 그룹 생성 (이 그룹을 복제해서 사용)
    const baseModelGroup = new THREE.Group();
    baseModelGroup.add(model);

    stocks.forEach(stock => {
      const charGroup = new THREE.Group();

      // 1. 3D 모델 추가
      const modelInstance = baseModelGroup.clone();
      charGroup.add(modelInstance);

      // 2. 머리 위 종목 심볼 (원판)
      const faceGeo = new THREE.CircleGeometry(0.3, 32);
      const faceTexture = createLogoTexture(stock.symbol, stock.color);
      const faceMat = new THREE.MeshBasicMaterial({ map: faceTexture, transparent: true, side: THREE.DoubleSide });
      const face = new THREE.Mesh(faceGeo, faceMat);
      face.position.set(0, targetHeight + 0.3, 0); // 캐릭터 머리 위로 띄움
      charGroup.add(face);

      // 3. 단상 위에 캐릭터 배치
      charGroup.position.set(stock.pos, stock.height - 0.5, 0);
      charGroup.userData = stock;

      // 4. 호버 효과를 위한 투명한 Hitbox (충돌 박스)
      const hitboxGeo = new THREE.CylinderGeometry(0.8, 0.8, targetHeight + 0.8);
      const hitboxMat = new THREE.MeshBasicMaterial({ visible: false });
      const hitbox = new THREE.Mesh(hitboxGeo, hitboxMat);
      hitbox.position.y = (targetHeight + 0.8) / 2;
      hitbox.userData = stock;
      charGroup.add(hitbox);

      characterMeshes.push(hitbox);
      scene.add(charGroup);

      // 등장 애니메이션
      charGroup.scale.set(0, 0, 0);
      gsap.to(charGroup.scale, {
        x: 1, y: 1, z: 1,
        duration: 1,
        delay: 0.5 + Math.random() * 0.5,
        ease: "elastic.out(1, 0.5)"
      });
    });
  }, undefined, (error) => {
    console.error('An error happened while loading the model:', error);
  });
}

// Raycaster for Hover
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
let hoveredStock = null;

const tooltip = document.getElementById('tooltip');
const tooltipName = document.getElementById('tooltip-name');
const tooltipPrice = document.getElementById('tooltip-price');
const tooltipChange = document.getElementById('tooltip-change');
const tooltipLogo = document.getElementById('tooltip-logo');

window.addEventListener('mousemove', (event) => {
  // Normalize mouse coordinates
  mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

  // Update tooltip position
  if (!tooltip.classList.contains('hidden')) {
    tooltip.style.left = (event.clientX + 15) + 'px';
    tooltip.style.top = (event.clientY + 15) + 'px';
  }
});

// Window Resize
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// Animation Loop
const clock = new THREE.Clock();

function tick() {
  const elapsedTime = clock.getElapsedTime();

  // Rotate particles slowly
  particlesMesh.rotation.y = elapsedTime * 0.05;

  // Update controls
  controls.update();

  // Raycasting
  raycaster.setFromCamera(mouse, camera);
  const intersects = raycaster.intersectObjects(characterMeshes);

  if (intersects.length > 0) {
    const object = intersects[0].object;
    const stock = object.userData;

    if (hoveredStock !== stock.id) {
      hoveredStock = stock.id;
      document.body.style.cursor = 'pointer';

      // Update HTML Tooltip
      tooltipName.textContent = stock.name;
      tooltipPrice.textContent = stock.price;
      tooltipChange.textContent = stock.change;
      tooltipLogo.style.backgroundColor = stock.color;
      // We can create a simple text placeholder for logo in CSS
      tooltipLogo.innerHTML = `<span style="display:flex;align-items:center;justify-content:center;height:100%;color:white;font-weight:bold;font-size:14px;">${stock.symbol}</span>`;

      if (stock.change.startsWith('-')) {
        tooltipChange.className = 'negative';
      } else {
        tooltipChange.className = 'positive';
      }

      tooltip.classList.remove('hidden');

      // Animate hover effect on the character's group
      gsap.to(object.parent.scale, {
        x: 1.1, y: 1.1, z: 1.1,
        duration: 0.3,
        ease: "power2.out"
      });
    }
  } else {
    if (hoveredStock !== null) {
      // Find the group of the previously hovered stock and scale down
      const prevStockHitbox = characterMeshes.find(m => m.userData.id === hoveredStock);
      if (prevStockHitbox) {
        gsap.to(prevStockHitbox.parent.scale, {
          x: 1, y: 1, z: 1,
          duration: 0.3,
          ease: "power2.out"
        });
      }

      hoveredStock = null;
      document.body.style.cursor = 'default';
      tooltip.classList.add('hidden');
    }
  }

  renderer.render(scene, camera);
  window.requestAnimationFrame(tick);
}

tick();
