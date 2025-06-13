import react from "react"
import { animate, createDraggable, createSpring, createScope } from "animejs"
import { useEffect, useRef, useState } from "react"
import reactLogo from './logo.svg'

function Header(props) {
  // let title = props.title
  // return <h1>{title ? title : "Default Title"}</h1>
  return <h1>{props.title}</h1>
}

function HomePage() {
  const title = "Develop. Preview. Ship."
  const names = ['Ada Lovelace', 'Grace Hopper', 'Margaret Hamilton']

  const [likes, setLikes] = react.useState(0)
  function handleClick() {
    console.log("Increment by 1")
    setLikes(likes + 1)
  }

  return (
    <div>
      <Header title={title} />
      <ul>
        {names.map(name => (
          <li>{name}</li>
        ))}
      </ul>
      <button onClick={handleClick}>Like {(likes)}</button>
    </div>
  )
}

function Animation() {
  const root = useRef(null)
  const scope = useRef(null)
  const [ rotations, setRotations ] = useState(0)

  useEffect(() => {
    scope.current = createScope({ root }).add( self => {
      animate('.logo', {
        scale: [
          { to: 1.25, ease: 'inOut(3)', duration: 200 },
          { to: 1, ease: createSpring({ stiffness: 300 }) }
        ],
        loop: true,
        loopDelay: 250,
      })

      createDraggable('.logo', {
        container: [0, 0, 0, 0],
        releaeEase: createSpring({ stiffness: 200 })
      })

      self.add('rotateLogo', (i) => {
        animate('.logo', {
          rotate: i * 360,
          ease: 'out(4)',
          duration: 1500,
        })
      })
    })
    return () => scope.current.revert()
  }, [])

  const handleClick = () => {
    setRotations(prev => {
      const newRotations = prev + 1
      scope.current.methods.rotateLogo(newRotations)
      return newRotations
    })
  }

  return (
    <div ref={root}>
      <div className="large centered row">
        <img src={reactLogo} className="logo react" alt="React Logo" width="500" height="600"/>
      </div>
      <div className="medium row">
        <fieldset className="controls">
          <button onClick={handleClick}>rotations: {rotations}</button>
        </fieldset>
      </div>
    </div>
  )
}

function App() { return <div><HomePage /><Animation /></div>}

export default App